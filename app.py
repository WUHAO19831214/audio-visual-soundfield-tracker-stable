"""Streamlit interface for local visual counting and soundfield capture."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.audio_devices import (
    AudioDeviceError,
    check_audio_input_device,
    get_default_input_device,
    get_default_input_device_index,
    list_audio_input_devices,
    test_audio_input_device,
)
from src.camera_devices import list_available_cameras, read_camera_preview
from src.camera_processor import CameraProcessor
from src.config import LOG_DIR, OUTPUT_DIR as CONFIG_OUTPUT_DIR
from src.csv_logger import write_csv
from src.detector import Detector
from src.fusion import SynchronizedCaptureSession
from src.local_capture import (
    AudioLevelWorker,
    LocalAudioWorker,
    LocalCameraWorker,
    LocalFusionWorker,
    check_writable_directory,
)
from src.trajectory_visualizer import (
    choose_main_track,
    draw_tracks_on_frame,
    save_trajectory_blank,
    save_trajectory_overlay,
)


OUTPUT_DIR = Path("data/output")
UPLOAD_DIR = Path("data/uploads")
MODES = ["图片计数", "视频计数", "摄像头实时计数", "视觉—音频同步采集"]
TRACK_FIELDS = [
    "timestamp",
    "track_id",
    "center_x",
    "center_y",
    "bbox_width",
    "bbox_height",
]


@st.cache_resource
def get_detector(confidence_threshold: float, person_only: bool) -> Detector:
    return Detector(
        confidence_threshold=confidence_threshold,
        person_only=person_only,
    )


def render_download(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=f"download_{path.name}_{label}",
        )


def _browser_component_key(base: str, reset_generation: int) -> str:
    session_key = "browser_component_session_id"
    if session_key not in st.session_state:
        st.session_state[session_key] = uuid.uuid4().hex[:10]
    return f"{base}-v2-{st.session_state[session_key]}-{reset_generation}"


def image_counting_mode(detector: Detector) -> None:
    st.subheader("图片计数")
    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.info("上传一张包含人物的图片开始检测。")
        return

    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("无法读取该图片。")
        return

    processor = CameraProcessor(detector=detector)
    annotated, detections = processor.detect_image(image)
    st.metric("检测到的 person 数量", len(detections))
    st.caption(f"检测后端：{processor.backend_name}")
    if detector.fallback_reason:
        st.warning(detector.fallback_reason)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)


def _process_uploaded_video(
    uploaded,
    confidence_threshold: float,
    person_only: bool,
    export_trajectory_video: bool = True,
    export_trajectory_images: bool = True,
    trajectory_background: str = "首帧",
    show_all_tracks: bool = True,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = UPLOAD_DIR / f"{timestamp}_{Path(uploaded.name).name}"
    input_path.write_bytes(uploaded.getvalue())
    output_path = OUTPUT_DIR / f"video_{timestamp}_tracked.mp4"
    trajectory_video_path = OUTPUT_DIR / f"video_{timestamp}_with_trajectory.mp4"
    csv_path = OUTPUT_DIR / f"video_{timestamp}_trajectory.csv"
    trajectory_overlay_path = OUTPUT_DIR / f"video_{timestamp}_trajectory_overlay.png"
    trajectory_blank_path = OUTPUT_DIR / f"video_{timestamp}_trajectory_blank.png"

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("无法打开上传的视频。")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建输出视频，请检查本机 OpenCV 编码支持。")
    trajectory_writer = None
    if export_trajectory_video:
        trajectory_writer = cv2.VideoWriter(
            str(trajectory_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not trajectory_writer.isOpened():
            writer.release()
            capture.release()
            raise RuntimeError("无法创建带轨迹输出视频，请检查本机 OpenCV 编码支持。")

    processor = CameraProcessor(
        Detector(
            confidence_threshold=confidence_threshold,
            person_only=person_only,
        )
    )
    trajectory: list[dict] = []
    track_history: dict[int | str, list[dict]] = {}
    unique_ids: set[int] = set()
    max_persons = 0
    frame_index = 0
    first_frame = None
    last_frame = None
    progress = st.progress(0.0, text="正在处理视频...")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if first_frame is None:
                first_frame = frame.copy()
            last_frame = frame.copy()
            frame_timestamp = frame_index / fps
            annotated, tracks, primary = processor.process_frame(frame, frame_timestamp)
            writer.write(annotated)
            unique_ids.update(int(track["track_id"]) for track in tracks)
            max_persons = max(max_persons, len(tracks))
            for track in tracks:
                track_history.setdefault(track["track_id"], []).append(
                    {
                        "frame_index": frame_index,
                        "timestamp": frame_timestamp,
                        "center_x": track["center_x"],
                        "center_y": track["center_y"],
                        "bbox_width": track["bbox_width"],
                        "bbox_height": track["bbox_height"],
                        "confidence": track.get("confidence", ""),
                        "class_name": track.get("class_name", ""),
                    }
                )
            if primary is not None:
                trajectory.append(
                    {
                        "timestamp": frame_timestamp,
                        "track_id": primary["track_id"],
                        "center_x": primary["center_x"],
                        "center_y": primary["center_y"],
                        "bbox_width": primary["bbox_width"],
                        "bbox_height": primary["bbox_height"],
                    }
                )
            if trajectory_writer is not None:
                current_main = primary["track_id"] if primary is not None else None
                trajectory_frame = draw_tracks_on_frame(
                    annotated,
                    track_history,
                    current_frame_index=frame_index,
                    main_track_id=current_main,
                    show_all_tracks=show_all_tracks,
                )
                trajectory_writer.write(trajectory_frame)
            frame_index += 1
            if frame_index % 5 == 0:
                progress.progress(
                    min(frame_index / frame_count, 1.0), text="正在处理视频..."
                )
    finally:
        capture.release()
        writer.release()
        if trajectory_writer is not None:
            trajectory_writer.release()
        progress.empty()

    write_csv(csv_path, TRACK_FIELDS, trajectory)
    main_track_id = choose_main_track(track_history)
    saved_overlay = None
    saved_blank = None
    if export_trajectory_images:
        background_frame = first_frame if trajectory_background == "首帧" else last_frame
        if background_frame is not None:
            saved_overlay = save_trajectory_overlay(
                background_frame,
                track_history,
                trajectory_overlay_path,
                main_track_id=main_track_id,
                show_all_tracks=show_all_tracks,
            )
        saved_blank = save_trajectory_blank(
            track_history,
            trajectory_blank_path,
            image_size=(width, height),
            main_track_id=main_track_id,
            show_all_tracks=show_all_tracks,
        )
    return {
        "video": output_path,
        "trajectory_video": trajectory_video_path if export_trajectory_video else None,
        "csv": csv_path,
        "trajectory_overlay": saved_overlay,
        "trajectory_blank": saved_blank,
        "main_track_id": main_track_id,
        "unique_count": len(unique_ids),
        "max_persons": max_persons,
        "backend": processor.backend_name,
        "tracking_warning": getattr(processor, "tracking_warning", ""),
    }


def video_counting_mode(confidence_threshold: float, person_only: bool) -> None:
    st.subheader("视频计数与轨迹记录")
    with st.expander("轨迹导出设置", expanded=True):
        export_trajectory_video = st.checkbox("导出带轨迹视频", value=True)
        export_trajectory_images = st.checkbox("导出轨迹 PNG 图片", value=True)
        trajectory_background = st.selectbox("轨迹叠加背景", ["首帧", "末帧"], index=0)
        show_all_tracks = st.checkbox("显示所有 track 轨迹", value=True)
    uploaded = st.file_uploader("上传视频", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is not None and st.button("开始处理视频", type="primary"):
        try:
            st.session_state.video_result = _process_uploaded_video(
                uploaded,
                confidence_threshold,
                person_only,
                export_trajectory_video=export_trajectory_video,
                export_trajectory_images=export_trajectory_images,
                trajectory_background=trajectory_background,
                show_all_tracks=show_all_tracks,
            )
        except Exception as exc:
            st.error(f"视频处理失败：{exc}")

    result = st.session_state.get("video_result")
    if not result:
        return
    first, second = st.columns(2)
    first.metric("累计出现的 track_id", result["unique_count"])
    second.metric("单帧最大人数", result["max_persons"])
    if result.get("main_track_id") is not None:
        st.caption(f"主轨迹 track_id：{result['main_track_id']}")
    st.caption(f"检测后端：{result['backend']}")
    if result.get("tracking_warning"):
        st.warning(result["tracking_warning"])
    st.video(str(result["video"]))
    render_download(result["video"], "下载标注视频", "video/mp4")
    if result.get("trajectory_video"):
        st.video(str(result["trajectory_video"]))
        render_download(result["trajectory_video"], "下载带轨迹视频", "video/mp4")
    if result.get("trajectory_overlay"):
        st.image(str(result["trajectory_overlay"]), caption="轨迹叠加图", use_container_width=True)
        render_download(result["trajectory_overlay"], "下载轨迹叠加 PNG", "image/png")
    if result.get("trajectory_blank"):
        st.image(str(result["trajectory_blank"]), caption="空白背景轨迹图", use_container_width=True)
        render_download(result["trajectory_blank"], "下载空白轨迹 PNG", "image/png")
    render_download(result["csv"], "下载轨迹 CSV", "text/csv")


def _camera_index_controls(prefix: str) -> int:
    scan_key = f"{prefix}_camera_scan"
    if scan_key not in st.session_state:
        with st.spinner("正在读取本机摄像头列表..."):
            st.session_state[scan_key] = list_available_cameras(max_index=5)

    st.markdown("**本机 OpenCV 摄像头**")
    st.caption(
        "Mac 内置摄像头通常是 Camera 0；iPhone Continuity Camera 更推荐用上面的浏览器模式。"
    )
    if st.button("刷新本机摄像头列表", key=f"{prefix}_scan_button"):
        with st.spinner("正在逐个测试摄像头编号..."):
            st.session_state[scan_key] = list_available_cameras(max_index=5)

    devices = st.session_state.get(scan_key, [])
    names = {item["index"]: item["name"] for item in devices}
    available_indexes = [item["index"] for item in devices]
    options = available_indexes or list(range(6))
    camera_index = st.selectbox(
        "摄像头 index",
        options=options,
        format_func=lambda index: (
            f"{names.get(index, f'Camera {index}')}（可用）"
            if index in available_indexes
            else f"Camera {index}（未确认）"
        ),
        key=f"{prefix}_camera_index",
    )
    if devices:
        st.success("可用摄像头：" + "、".join(item["name"] for item in devices))
    else:
        st.warning("暂未确认可用 OpenCV 摄像头。请检查 macOS 摄像头权限后点“刷新本机摄像头列表”。")
    if st.button("读取一帧预览", key=f"{prefix}_test_button"):
        try:
            frame = read_camera_preview(int(camera_index))
            st.session_state[f"{prefix}_preview"] = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            st.success(f"Camera {camera_index} 可用")
        except Exception as exc:
            st.error(f"摄像头测试失败：{exc}")
    preview = st.session_state.get(f"{prefix}_preview")
    if preview is not None:
        st.image(preview, caption=f"Camera {camera_index} 单帧预览", width=480)
    return int(camera_index)


@st.fragment(run_every=0.2)
def _render_local_camera_worker(worker_key: str) -> None:
    worker = st.session_state.get(worker_key)
    if worker is None:
        return
    frame = worker.latest_frame()
    status = worker.status()
    if frame is not None:
        st.image(frame, channels="RGB", use_container_width=True)
    left, right = st.columns(2)
    left.metric("当前人数", status.get("count", 0))
    primary = status.get("primary")
    right.metric("主 track_id", primary["track_id"] if primary else "--")
    st.caption(f"检测后端：{status['backend']}")
    if status.get("tracking_warning"):
        st.warning(status["tracking_warning"])
    if status.get("error"):
        st.error(status["error"])


def _browser_camera_counting(
    confidence_threshold: float, person_only: bool
) -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from src.browser_capture import LiveCountingProcessor

    st.info(
        "如果要使用 iPhone Continuity Camera，请先允许浏览器使用默认摄像头，"
        "停止采集后点击 WebRTC 组件中的 Select Device，再选择名称包含 iPhone 的摄像头。"
    )
    with st.expander("浏览器看不到摄像头或 iPhone？"):
        st.markdown(
            "- 使用最新版 Chrome 或 Safari，并确认地址是 `localhost`。\n"
            "- 点击地址栏摄像头权限，先允许默认摄像头，再刷新页面。\n"
            "- 确认 Mac 与 iPhone 使用同一 Apple ID，Wi-Fi、蓝牙和接力已开启。\n"
            "- 浏览器设备列表由浏览器权限控制，Python 无法直接读取。"
        )
    reset_key = "camera_browser_device_reset"
    if st.button("重置浏览器设备选择", key="reset_browser_camera"):
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        st.info("已清除本页面使用的旧设备选择。请重新点击 Start 并授权。")
    component_generation = st.session_state.get(reset_key, 0)
    try:
        context = webrtc_streamer(
            key=_browser_component_key("camera-counting", component_generation),
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: LiveCountingProcessor(
                confidence_threshold, person_only
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": False,
            },
            async_processing=True,
        )
    except Exception as exc:
        st.error(f"浏览器摄像头组件启动失败：{exc}")
        st.info("请切换到“OpenCV 本机摄像头（推荐）”继续使用。")
        return
    count_slot = st.empty()
    track_slot = st.empty()
    while context.state.playing:
        processor = context.video_processor
        if processor is not None:
            status = processor.status()
            count_slot.metric("当前人数", status["count"])
            primary = status["primary"]
            track_slot.info(
                f"当前主 track_id：{primary['track_id']}"
                if primary
                else "当前未检测到 person"
            )
        time.sleep(0.2)


def camera_counting_mode(confidence_threshold: float, person_only: bool) -> None:
    st.subheader("摄像头实时计数")
    source = st.radio(
        "采集方式",
        [
            "浏览器摄像头（推荐用于 iPhone Continuity Camera）",
            "OpenCV 本机摄像头（推荐用于本地稳定实验）",
        ],
        horizontal=True,
        key="counting_capture_source",
    )
    if source.startswith("浏览器"):
        _browser_camera_counting(confidence_threshold, person_only)
        return

    camera_index = _camera_index_controls("counting")
    worker_key = "local_counting_worker"
    worker = st.session_state.get(worker_key)
    start_col, stop_col = st.columns(2)
    if start_col.button("开始实时检测", type="primary", disabled=bool(worker and worker.running)):
        try:
            if worker is not None:
                worker.stop()
            worker = LocalCameraWorker(
                camera_index,
                confidence_threshold=confidence_threshold,
                person_only=person_only,
            )
            worker.start()
            st.session_state[worker_key] = worker
        except Exception as exc:
            st.error(f"无法开始本机摄像头检测：{exc}")
    if stop_col.button("停止实时检测", disabled=not bool(worker and worker.running)):
        worker.stop()
    _render_local_camera_worker(worker_key)


def _display_capture_artifacts(artifacts, acoustic_options: dict | None = None) -> None:
    summary = getattr(artifacts, "summary", None) or {}
    if summary and not summary.get("ok", True):
        st.warning(
            "采集已停止，但本次数据不完整："
            f"视觉样本 {summary.get('video_sample_count', 0)}，"
            f"音频样本 {summary.get('audio_sample_count', 0)}，"
            f"融合样本 {summary.get('fused_sample_count', 0)}。"
        )
        if int(summary.get("audio_sample_count", 0) or 0) == 0:
            st.error("本次采集没有收到音频样本，融合图中的声强和频率不可信。请重新测试麦克风后再次采集。")
    else:
        st.success("采集已停止，融合 CSV 和可视化文件已生成。")
    st.write(f"融合数据：`{artifacts.fused_csv}`")
    render_download(artifacts.fused_csv, "下载融合 CSV", "text/csv")
    summary_json = getattr(artifacts, "summary_json", None)
    if summary_json:
        render_download(summary_json, "下载采集 summary JSON", "application/json")

    plots = [
        (artifacts.trajectory_plot, "位置—时间轨迹"),
        (artifacts.intensity_plot, "空间声强分布"),
        (artifacts.frequency_plot, "空间主频分布"),
    ]
    for path, title in plots:
        if path and path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)

    trajectory_overlay = getattr(artifacts, "trajectory_overlay_plot", None)
    if trajectory_overlay and trajectory_overlay.exists():
        st.markdown("**融合轨迹叠加图**")
        st.image(str(trajectory_overlay), use_container_width=True)
        render_download(trajectory_overlay, "下载融合轨迹叠加 PNG", "image/png")

    acoustic_plots = {
        ("db", "values"): (
            getattr(artifacts, "acoustic_db_values_plot", None),
            "相对声强数值轨迹",
        ),
        ("db", "colormap"): (
            getattr(artifacts, "acoustic_db_colormap_plot", None),
            "相对声强颜色轨迹",
        ),
        ("dominant_frequency_hz", "values"): (
            getattr(artifacts, "acoustic_frequency_values_plot", None),
            "主频数值轨迹",
        ),
        ("dominant_frequency_hz", "colormap"): (
            getattr(artifacts, "acoustic_frequency_colormap_plot", None),
            "主频颜色轨迹",
        ),
        ("spectral_centroid_hz", "colormap"): (
            getattr(artifacts, "acoustic_spectral_centroid_colormap_plot", None),
            "频谱质心颜色轨迹",
        ),
    }
    existing_acoustic = [
        (path, title) for path, title in acoustic_plots.values() if path and path.exists()
    ]
    if existing_acoustic:
        st.info("同步采集当前保存 CSV 和 PNG；原始视频流尚未保存，所以暂不生成声学轨迹视频。")
        metric = (acoustic_options or {}).get("metric", "db")
        display = (acoustic_options or {}).get("display", "both")
        selected_keys = []
        if display in ("values", "both"):
            selected_keys.append((metric, "values"))
        if display in ("colormap", "both"):
            selected_keys.append((metric, "colormap"))
        if metric == "spectral_centroid_hz" and display in ("values", "both"):
            selected_keys.append((metric, "colormap"))

        for key in selected_keys:
            path, title = acoustic_plots.get(key, (None, ""))
            if path and path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)
                render_download(path, f"下载{title} PNG", "image/png")

        with st.expander("全部声学轨迹 PNG"):
            for path, title in existing_acoustic:
                render_download(path, f"下载{title}", "image/png")

    with st.expander("原始数据文件"):
        render_download(artifacts.visual_csv, "下载视觉轨迹 CSV", "text/csv")
        render_download(artifacts.audio_csv, "下载音频特征 CSV", "text/csv")


def _browser_fusion_capture(
    confidence_threshold: float, person_only: bool, acoustic_options: dict | None
) -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from src.browser_capture import FusionVideoProcessor

    st.info("iPhone/Continuity Camera 用这条路径：浏览器选摄像头，本页选麦克风。")
    with st.expander("看不到 iPhone 摄像头？"):
        st.markdown(
            "- 先点 WebRTC 组件里的 `Start`，允许浏览器摄像头权限。\n"
            "- 再点 `Select Device`，从浏览器设备列表里选 iPhone。\n"
            "- 如果列表没有 iPhone，确认同一 Apple ID、Wi-Fi、蓝牙和接力已开启，然后刷新页面。"
        )
    reset_key = "fusion_browser_device_reset"
    if st.button("重置浏览器摄像头选择", key="reset_browser_fusion"):
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        st.info("已清除本页面使用的旧摄像头选择。请重新点击 Start 并授权。")
    component_generation = st.session_state.get(reset_key, 0)

    st.markdown("**切换路径**")
    st.markdown(
        "1. 点击下方 WebRTC 组件的 **Start**。\n"
        "2. 点击组件里的 **Select Device**，选择 iPhone 或其他浏览器可见摄像头。\n"
        "3. 麦克风在下面选择，推荐 `Wireless Mic Rx` 或系统默认麦克风。"
    )

    audio_choice = _audio_device_selector(key="browser_audio_device")
    local_audio_device_index = None if audio_choice is None else int(audio_choice)
    st.caption(
        "当前采集路径：浏览器只负责摄像头；麦克风使用 sounddevice 设备 "
        f"{st.session_state.get('browser_audio_device_selected_audio_input_name', '')} "
        f"| index={local_audio_device_index}"
    )
    audio_worker_key = "browser_fusion_audio_worker"
    existing_audio_worker = st.session_state.get(audio_worker_key)
    _microphone_diagnostics(
        "browser_audio_device",
        local_audio_device_index,
        existing_audio_worker.status() if existing_audio_worker is not None else None,
    )

    session = st.session_state.get("fusion_session")
    if session is None or session.finalized:
        session = SynchronizedCaptureSession(output_dir=OUTPUT_DIR)
        st.session_state.fusion_session = session

    try:
        webrtc_options = {
            "key": _browser_component_key(
                "audio-visual-fusion", component_generation
            ),
            "mode": WebRtcMode.SENDRECV,
            "video_processor_factory": lambda: FusionVideoProcessor(
                session, confidence_threshold, person_only
            ),
            "media_stream_constraints": {
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": False,
            },
            "sendback_audio": False,
            "async_processing": True,
        }
        context = webrtc_streamer(**webrtc_options)
    except Exception as exc:
        st.error(f"浏览器音视频组件启动失败：{exc}")
        st.info("请切换到“OpenCV 摄像头 + sounddevice 麦克风（推荐）”。")
        return

    columns = st.columns(4)
    track_slot = columns[0].empty()
    position_slot = columns[1].empty()
    db_slot = columns[2].empty()
    frequency_slot = columns[3].empty()
    sample_slot = st.empty()

    was_playing = context.state.playing
    audio_worker = st.session_state.get(audio_worker_key)
    if was_playing:
        worker_needs_restart = (
            audio_worker is None
            or audio_worker.session is not session
            or audio_worker.audio_device_index != local_audio_device_index
            or not audio_worker.running
        )
        if worker_needs_restart:
            if audio_worker is not None:
                audio_worker.stop()
            _stop_audio_level_monitor("browser_audio_device")
            test_result = test_audio_input_device(local_audio_device_index, duration=1.0)
            st.session_state["browser_audio_device_last_audio_test"] = test_result
            if not test_result.get("ok"):
                st.session_state.fusion_audio_start_error = test_result.get("error", "")
                st.error("麦克风预检失败：" + st.session_state.fusion_audio_start_error)
                return
            try:
                audio_worker = LocalAudioWorker(local_audio_device_index, session)
                audio_worker.start(require_ready=True, timeout_sec=2.0)
                st.session_state[audio_worker_key] = audio_worker
                st.session_state.fusion_capture_active = True
                st.success("音频流已就绪，开始视觉—音频同步采集。")
            except Exception as exc:
                st.session_state.fusion_audio_start_error = str(exc)
                st.error(str(exc))
                return
        else:
            st.session_state.fusion_capture_active = True
        if audio_worker and audio_worker.audio_error:
            st.warning("本机麦克风采集异常：" + audio_worker.audio_error)
    elif audio_worker is not None and audio_worker.running:
        audio_worker.stop()

    while context.state.playing:
        status = session.status()
        visual = status["visual"]
        audio = status["audio"]
        track_slot.metric("主 track_id", visual["track_id"] if visual else "--")
        position_slot.metric(
            "位置 (x, y)",
            f"{visual['center_x']:.0f}, {visual['center_y']:.0f}" if visual else "--",
        )
        db_slot.metric("相对声强", f"{audio['db']:.1f} dB" if audio else "--")
        frequency_slot.metric(
            "主频", f"{audio['dominant_frequency_hz']:.1f} Hz" if audio else "--"
        )
        sample_slot.caption(
            f"视觉样本 {status['visual_samples']} 条｜音频样本 {status['audio_samples']} 条｜"
            f"已采集 {status['elapsed_sec']:.1f} 秒"
        )
        time.sleep(0.1)

    should_finalize = (
        not context.state.playing
        and not session.finalized
        and (was_playing or st.session_state.get("fusion_capture_active", False))
    )
    if should_finalize:
        with st.spinner("正在对齐时间戳并生成图像..."):
            audio_worker = st.session_state.get(audio_worker_key)
            if audio_worker is not None:
                audio_worker.stop()
            if not session.audio_snapshot():
                st.error(
                    "麦克风设备已选择，但正式采集没有收到音频样本，"
                    "本次不生成成功结果。请点击“测试当前麦克风 1 秒”或切换设备后重试。"
                )
                st.session_state.fusion_capture_active = False
                return
            artifacts = session.finalize(
                export_acoustic_trajectory=bool(
                    (acoustic_options or {}).get("export", True)
                ),
                acoustic_label_every=int(
                    (acoustic_options or {}).get("label_every", 10)
                ),
            )
            st.session_state.last_fusion_artifacts = artifacts
            st.session_state.fusion_capture_active = False

    artifacts = st.session_state.get("last_fusion_artifacts")
    if artifacts is not None:
        _display_capture_artifacts(artifacts, acoustic_options)


def _audio_meter_html(meter: float) -> str:
    block_count = 16
    active_count = int(round(max(0.0, min(1.0, meter)) * block_count))
    blocks = []
    for index in range(block_count):
        color = "#c8d8ce" if index < active_count else "rgba(255,255,255,0.12)"
        blocks.append(
            "<span style='display:inline-block;width:7px;height:18px;"
            f"border-radius:5px;background:{color};margin-right:9px;'></span>"
        )
    return (
        "<div style='display:flex;align-items:center;gap:18px;"
        "padding:8px 0 2px 0;'>"
        "<div style='min-width:82px;font-weight:600;'>输入电平</div>"
        "<div style='display:flex;align-items:center;'>"
        + "".join(blocks)
        + "</div></div>"
    )


@st.fragment(run_every=0.7)
def _render_audio_input_level(device_index: int | None, key: str) -> None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        st.markdown(_audio_meter_html(0.0), unsafe_allow_html=True)
        st.caption("测试环境中不打开真实麦克风。")
        return

    monitor_key = f"{key}_monitor"
    monitor = st.session_state.get(monitor_key)
    local_worker = st.session_state.get("local_fusion_worker")
    browser_worker = st.session_state.get("browser_fusion_audio_worker")
    if (
        local_worker is not None
        and getattr(local_worker, "running", False)
        or browser_worker is not None
        and getattr(browser_worker, "running", False)
    ):
        if monitor is not None:
            monitor.stop()
        st.markdown(_audio_meter_html(0.0), unsafe_allow_html=True)
        st.caption("采集中：实时声强请看下方采集指标。")
        return

    if (
        monitor is None
        or monitor.audio_device_index != device_index
        or not monitor.running
    ):
        if monitor is not None:
            monitor.stop()
        monitor = AudioLevelWorker(device_index)
        monitor.start()
        st.session_state[monitor_key] = monitor

    if monitor.error:
        st.markdown(_audio_meter_html(0.0), unsafe_allow_html=True)
        st.caption(monitor.error)
        return

    level = monitor.status()
    st.markdown(_audio_meter_html(float(level["meter"])), unsafe_allow_html=True)
    st.caption(
        f"{level['name']}｜{level['dbfs']:.1f} dBFS｜峰值 {level['peak']:.3f}"
    )


def _audio_device_selector(key: str = "local_audio_device") -> int | None:
    try:
        default_device = get_default_input_device()
        devices = list_audio_input_devices()
    except AudioDeviceError as exc:
        st.warning(str(exc))
        return None

    refresh_col, _ = st.columns([1, 1])
    if refresh_col.button("刷新麦克风列表", key=f"{key}_refresh"):
        st.rerun()
    st.caption(
        "如果刚插入 USB 麦克风后仍未出现，请重启 Streamlit；"
        "macOS/PortAudio 有时会缓存启动时的设备列表。"
    )

    options: list[int | None] = [None] + [item["index"] for item in devices]
    device_names = {item["index"]: item["name"] for item in devices}
    device_details = {
        item["index"]: (
            f"{item['name']}｜{item['max_input_channels']}ch｜"
            f"{int(item['default_samplerate'])} Hz"
        )
        for item in devices
    }
    default_text = (
        f"默认麦克风（{default_device['name']}）"
        if default_device
        else "默认麦克风"
    )
    selected = st.selectbox(
        "麦克风输入设备",
        options=options,
        format_func=lambda value: default_text
        if value is None
        else f"{device_names.get(value, '未知设备')} | index={value} | "
        f"{int(next((item['default_samplerate'] for item in devices if item['index'] == value), 0))} Hz",
        key=key,
    )
    if devices:
        with st.expander("当前 sounddevice 可见输入设备"):
            for item in devices:
                st.write(f"{item['index']}: {device_details[item['index']]}")
    else:
        st.warning("sounddevice 当前没有枚举到输入设备。")

    selected_index = (
        None
        if selected is None and default_device is None
        else int(default_device["index"])
        if selected is None
        else int(selected)
    )
    selected_name = (
        default_device["name"]
        if selected is None and default_device
        else device_names.get(selected_index, "默认麦克风")
    )
    st.session_state["selected_audio_input_index"] = selected_index
    st.session_state["selected_audio_input_name"] = selected_name
    st.session_state[f"{key}_selected_audio_input_index"] = selected_index
    st.session_state[f"{key}_selected_audio_input_name"] = selected_name
    st.caption(
        f"正式采集将使用 sounddevice 设备：{selected_name} | index={selected_index}"
    )
    _render_audio_input_level(selected_index, key=f"{key}_level")
    return selected_index


def _format_audio_time(value) -> str:
    if not value:
        return "--"
    try:
        return datetime.fromtimestamp(float(value)).strftime("%H:%M:%S")
    except Exception:
        return str(value)


def _stop_audio_level_monitor(selector_key: str) -> None:
    monitor_key = f"{selector_key}_level_monitor"
    monitor = st.session_state.get(monitor_key)
    if monitor is not None:
        monitor.stop()
        st.session_state[monitor_key] = None


def _reset_audio_capture_state() -> None:
    for key in ("local_audio_device", "browser_audio_device"):
        _stop_audio_level_monitor(key)
    browser_worker = st.session_state.get("browser_fusion_audio_worker")
    if browser_worker is not None:
        browser_worker.stop()
        st.session_state["browser_fusion_audio_worker"] = None
    local_worker = st.session_state.get("local_fusion_worker")
    if local_worker is not None and not getattr(local_worker, "running", False):
        st.session_state["local_fusion_worker"] = None
    st.session_state["fusion_audio_start_error"] = ""


def _microphone_diagnostics(
    selector_key: str,
    selected_index: int | None,
    capture_status: dict | None = None,
) -> None:
    with st.expander("麦克风诊断", expanded=False):
        try:
            devices = list_audio_input_devices()
            default_index, default_note = get_default_input_device_index()
        except AudioDeviceError as exc:
            st.warning(str(exc))
            devices = []
            default_index, default_note = None, ""

        selected_name = st.session_state.get(f"{selector_key}_selected_audio_input_name", "")
        st.write(f"当前选择设备：`{selected_name or '--'}`")
        st.write(f"当前选择 device index：`{selected_index}`")
        st.write(f"当前默认 input device：`{default_index}` {default_note}")
        if devices:
            st.caption("当前可见输入设备")
            for item in devices:
                st.text(
                    f"{item['index']}: {item['name']} | "
                    f"{item['max_input_channels']}ch | "
                    f"{int(item['default_samplerate'])} Hz"
                )
        else:
            st.warning("sounddevice 当前没有枚举到输入设备。")

        test_key = f"{selector_key}_last_audio_test"
        button_col, reset_col = st.columns(2)
        if button_col.button("测试当前麦克风 1 秒", key=f"{selector_key}_test_audio"):
            _stop_audio_level_monitor(selector_key)
            result = test_audio_input_device(selected_index, duration=1.0)
            st.session_state[test_key] = result
        if reset_col.button("重新初始化麦克风状态", key=f"{selector_key}_reset_audio"):
            _reset_audio_capture_state()
            st.success("已关闭旧音频流并清理麦克风状态。")

        test_result = st.session_state.get(test_key)
        if test_result:
            if test_result.get("ok"):
                st.success(
                    "最近测试成功："
                    f"{test_result['device_name']} | index={test_result['device_index']} | "
                    f"{test_result['sample_rate']} Hz | "
                    f"RMS {test_result['rms']:.5f} | {test_result['db']:.1f} dBFS"
                )
            else:
                st.error("最近测试失败：" + str(test_result.get("error", "")))

        if capture_status:
            st.markdown("**正式采集音频状态**")
            st.write(f"实际传给 sounddevice 的 index：`{capture_status.get('device_index')}`")
            st.write(f"音频流已启动：`{capture_status.get('stream_started')}`")
            st.write(f"正在接收数据：`{capture_status.get('receiving')}`")
            st.write(f"音频 chunk 数：`{capture_status.get('chunk_count', 0)}`")
            st.write(f"最近 chunk 时间：`{_format_audio_time(capture_status.get('last_chunk_at'))}`")
            st.write(f"当前 RMS / dB：`{capture_status.get('last_rms', 0.0):.5f}` / `{capture_status.get('last_db', -120.0):.1f}`")
            if capture_status.get("last_error"):
                st.warning("最近一次错误：" + str(capture_status["last_error"]))


def _run_local_preflight(camera_index: int, audio_device_index: int | None) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    try:
        frame = read_camera_preview(camera_index)
        checks.append(
            ("摄像头", True, f"Camera {camera_index}: {frame.shape[1]}x{frame.shape[0]}")
        )
    except Exception as exc:
        checks.append(("摄像头", False, str(exc)))

    try:
        audio = check_audio_input_device(audio_device_index)
        checks.append(("麦克风", True, f"{audio['name']} @ {audio['sample_rate']} Hz"))
    except AudioDeviceError as exc:
        checks.append(("麦克风", False, str(exc)))

    detector = Detector()
    checks.append(
        (
            "检测器",
            detector.backend_name == "YOLO",
            str(detector.weight_path) if detector.weight_path else detector.fallback_reason,
        )
    )
    for label, path in (("输出目录", CONFIG_OUTPUT_DIR), ("日志目录", LOG_DIR)):
        ok, detail = check_writable_directory(path)
        checks.append((label, ok, detail))
    return checks


def _show_preflight(checks: list[tuple[str, bool, str]]) -> None:
    for label, ok, detail in checks:
        if ok:
            st.success(f"{label}：{detail}")
        else:
            st.warning(f"{label}：{detail}")


@st.fragment(run_every=0.2)
def _render_local_fusion_worker(worker_key: str) -> None:
    worker = st.session_state.get(worker_key)
    if worker is None:
        return
    frame = worker.latest_frame()
    status = worker.status()
    if frame is not None:
        st.image(frame, channels="RGB", use_container_width=True)
    columns = st.columns(4)
    visual = status.get("visual")
    audio = status.get("audio")
    columns[0].metric("主 track_id", visual["track_id"] if visual else "--")
    columns[1].metric(
        "位置 (x, y)",
        f"{visual['center_x']:.0f}, {visual['center_y']:.0f}" if visual else "--",
    )
    columns[2].metric("相对声强", f"{audio['db']:.1f} dB" if audio else "--")
    columns[3].metric(
        "主频", f"{audio['dominant_frequency_hz']:.1f} Hz" if audio else "--"
    )
    st.caption(
        f"检测后端：{status['backend']}｜视觉样本 {status['visual_samples']} 条｜"
        f"音频样本 {status['audio_samples']} 条｜已采集 {status['elapsed_sec']:.1f} 秒"
    )
    audio_capture = status.get("audio_capture", {})
    if audio_capture:
        st.caption(
            "音频流："
            f"index={audio_capture.get('device_index')}｜"
            f"started={audio_capture.get('stream_started')}｜"
            f"receiving={audio_capture.get('receiving')}｜"
            f"chunks={audio_capture.get('chunk_count', 0)}｜"
            f"last={_format_audio_time(audio_capture.get('last_chunk_at'))}｜"
            f"RMS={audio_capture.get('last_rms', 0.0):.5f}｜"
            f"dB={audio_capture.get('last_db', -120.0):.1f}"
        )
    if status.get("audio_error"):
        st.warning(
            "麦克风未正常采集，但摄像头仍在运行。错误：" + status["audio_error"]
        )
    if status.get("tracking_warning"):
        st.warning(status["tracking_warning"])
    if status.get("error"):
        st.error(status["error"])


def _local_fusion_capture(
    confidence_threshold: float, person_only: bool, acoustic_options: dict | None
) -> None:
    st.info("本机固定实验用这条路径：下方选择 OpenCV 摄像头和 sounddevice 麦克风。")
    camera_index = _camera_index_controls("fusion")
    audio_device_index = _audio_device_selector(key="local_audio_device")
    worker_key = "local_fusion_worker"
    worker = st.session_state.get(worker_key)
    _microphone_diagnostics(
        "local_audio_device",
        audio_device_index,
        worker.audio_capture.status() if worker is not None else None,
    )
    if st.button("运行采集前检查"):
        with st.spinner("正在检查摄像头、麦克风、检测器和输出目录..."):
            checks = _run_local_preflight(camera_index, audio_device_index)
            st.session_state.local_preflight_checks = checks
    checks = st.session_state.get("local_preflight_checks")
    if checks:
        with st.expander("采集前检查结果", expanded=True):
            _show_preflight(checks)

    start_col, stop_col = st.columns(2)
    if start_col.button(
        "开始同步采集", type="primary", disabled=bool(worker and worker.running)
    ):
        checks = _run_local_preflight(camera_index, audio_device_index)
        st.session_state.local_preflight_checks = checks
        camera_ok = next(ok for label, ok, _ in checks if label == "摄像头")
        output_ok = all(
            ok for label, ok, _ in checks if label in ("输出目录", "日志目录")
        )
        if not camera_ok or not output_ok:
            st.error("摄像头或输出目录检查失败，未开始采集。请查看检查结果。")
        else:
            try:
                _stop_audio_level_monitor("local_audio_device")
                audio_test = test_audio_input_device(audio_device_index, duration=1.0)
                st.session_state["local_audio_device_last_audio_test"] = audio_test
                if not audio_test.get("ok"):
                    st.error("麦克风预检失败：" + str(audio_test.get("error", "")))
                    return
                if worker is not None:
                    worker.stop()
                worker = LocalFusionWorker(
                    camera_index=camera_index,
                    audio_device_index=audio_device_index,
                    output_dir=OUTPUT_DIR,
                    confidence_threshold=confidence_threshold,
                    person_only=person_only,
                )
                worker.start()
                st.session_state[worker_key] = worker
                st.success("音频流已就绪，开始视觉—音频同步采集。")
            except Exception as exc:
                st.error(f"无法开始同步采集：{exc}")

    can_finalize = bool(worker and not worker.session.finalized)
    if stop_col.button("停止并生成结果", disabled=not can_finalize):
        try:
            with st.spinner("正在停止设备、融合时间戳并生成图像..."):
                artifacts = worker.stop_and_finalize(
                    export_acoustic_trajectory=bool(
                        (acoustic_options or {}).get("export", True)
                    ),
                    acoustic_label_every=int(
                        (acoustic_options or {}).get("label_every", 10)
                    ),
                )
                st.session_state.last_fusion_artifacts = artifacts
        except Exception as exc:
            st.error(f"停止或导出失败：{exc}")

    _render_local_fusion_worker(worker_key)
    artifacts = st.session_state.get("last_fusion_artifacts")
    if artifacts is not None:
        _display_capture_artifacts(artifacts, acoustic_options)


def _acoustic_export_controls() -> dict:
    with st.expander("声学轨迹导出设置", expanded=True):
        export = st.checkbox("导出声学轨迹 PNG", value=True)
        metric = st.selectbox(
            "优先显示指标",
            options=["db", "dominant_frequency_hz", "spectral_centroid_hz"],
            format_func=lambda value: {
                "db": "相对声强 dB",
                "dominant_frequency_hz": "主频 Hz",
                "spectral_centroid_hz": "频谱质心 Hz",
            }[value],
        )
        display = st.selectbox(
            "优先显示方式",
            options=["values", "colormap", "both"],
            format_func=lambda value: {
                "values": "数值标注",
                "colormap": "颜色热力",
                "both": "两者都显示",
            }[value],
        )
        label_every = st.number_input(
            "数值标注间隔（每 N 个匹配点标一次）",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
        )
    return {
        "export": export,
        "metric": metric,
        "display": display,
        "label_every": int(label_every),
    }


def fusion_capture_mode(confidence_threshold: float, person_only: bool) -> None:
    st.subheader("视觉—音频同步采集")
    acoustic_options = _acoustic_export_controls()
    source = st.radio(
        "采集方式",
        [
            "浏览器摄像头 / 麦克风（推荐用于 iPhone Continuity Camera）",
            "OpenCV 摄像头 + sounddevice 麦克风（推荐用于本地稳定实验）",
        ],
        horizontal=True,
        key="fusion_capture_source",
    )
    if source.startswith("浏览器"):
        _browser_fusion_capture(confidence_threshold, person_only, acoustic_options)
    else:
        _local_fusion_capture(confidence_threshold, person_only, acoustic_options)


def _stop_inactive_local_workers(mode: str) -> None:
    counting_worker = st.session_state.get("local_counting_worker")
    if mode != "摄像头实时计数" and counting_worker and counting_worker.running:
        counting_worker.stop()

    fusion_worker = st.session_state.get("local_fusion_worker")
    if (
        mode != "视觉—音频同步采集"
        and fusion_worker
        and not fusion_worker.session.finalized
    ):
        st.session_state.last_fusion_artifacts = fusion_worker.stop_and_finalize()

    browser_audio_worker = st.session_state.get("browser_fusion_audio_worker")
    if mode != "视觉—音频同步采集" and browser_audio_worker:
        browser_audio_worker.stop()

    if mode != "视觉—音频同步采集":
        for key, value in list(st.session_state.items()):
            if key.endswith("_level_monitor") and value is not None:
                value.stop()


def _render_device_diagnostics(detector: Detector) -> None:
    with st.sidebar.expander("设备诊断"):
        st.write(f"检测后端：**{detector.backend_name}**")
        st.caption(
            "YOLO 权重路径："
            + (str(detector.weight_path) if detector.weight_path else "未加载")
        )

        if st.button("扫描 OpenCV 摄像头 0–5", key="diagnostic_camera_scan"):
            with st.spinner("正在扫描摄像头编号..."):
                st.session_state.diagnostic_cameras = list_available_cameras(
                    max_index=5
                )
        cameras = st.session_state.get("diagnostic_cameras")
        if cameras is None:
            st.caption("OpenCV 摄像头：尚未扫描")
        elif cameras:
            st.caption(
                "OpenCV 可用 index："
                + "、".join(str(item["index"]) for item in cameras)
            )
        else:
            st.warning("OpenCV 未扫描到摄像头；这不代表浏览器无法使用 iPhone。")

        try:
            audio_devices = list_audio_input_devices()
            if audio_devices:
                st.caption("sounddevice 输入设备：")
                for item in audio_devices:
                    st.text(f"{item['index']}: {item['name']}")
            else:
                st.caption("sounddevice 未发现输入设备")
        except AudioDeviceError as exc:
            st.warning(str(exc))

        st.caption(
            "浏览器设备列表由浏览器权限控制，需先授权，再由 WebRTC 组件的 "
            "Select Device 显示。"
        )
        st.info(
            "推荐：iPhone Continuity Camera 使用浏览器模式；固定本地实验使用 "
            "OpenCV + sounddevice。"
        )


def main() -> None:
    st.set_page_config(page_title="视觉—声音空间分析", layout="wide")
    st.title("视觉—声音同步采集与空间声场分析系统")
    st.caption("本地实验原型｜图像平面坐标｜相对声强")

    mode = st.sidebar.radio("功能模式", MODES)
    _stop_inactive_local_workers(mode)
    st.sidebar.subheader("检测设置")
    confidence_threshold = st.sidebar.slider(
        "conf 阈值", min_value=0.1, max_value=0.9, value=0.25, step=0.05
    )
    person_only = st.sidebar.checkbox("只检测 person", value=True)
    detector = get_detector(confidence_threshold, person_only)
    st.sidebar.metric("检测后端", detector.backend_name)
    st.sidebar.caption(
        "YOLO 权重路径："
        + (str(detector.weight_path) if detector.weight_path else "未加载")
    )
    st.sidebar.caption("Tracker：bytetrack.yaml")
    if detector.backend_name != "YOLO":
        st.sidebar.warning(
            f"当前已回退到 OpenCV HOG。{detector.fallback_reason}"
        )
    _render_device_diagnostics(detector)

    if mode == "图片计数":
        image_counting_mode(detector)
    elif mode == "视频计数":
        video_counting_mode(confidence_threshold, person_only)
    elif mode == "摄像头实时计数":
        camera_counting_mode(confidence_threshold, person_only)
    else:
        fusion_capture_mode(confidence_threshold, person_only)


if __name__ == "__main__":
    main()
