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
    classify_input_device,
    diagnose_audio_devices,
    find_preferred_input_device,
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
from src.object_template_tracker import ObjectTemplateTracker, validate_bbox
from src.tennis_ball_tracker import (
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    TennisBallTracker,
    estimate_hsv_range_from_roi,
    make_tennis_mask,
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
YOLO_TRACKING_LABEL = "YOLO person（人形追踪）"
TENNIS_TRACKING_LABEL = "Tennis ball marker（网球标记追踪，推荐）"
CUSTOM_OBJECT_TRACKING_LABEL = "指定物体（模板追踪，实验性）"
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
        "iPhone 没有出现只是可选设备暂时不可见，不影响 FaceTime 高清相机正常采集。"
    )
    with st.expander("浏览器看不到摄像头或 iPhone？"):
        st.markdown(
            "- 使用最新版 Chrome 或 Safari，并确认地址是 `localhost`。\n"
            "- 点击地址栏摄像头权限，先允许默认摄像头，再刷新页面。\n"
            "- 确认 iPhone 已解锁并靠近 Mac，Wi-Fi、蓝牙和接力已开启。\n"
            "- 关闭可能占用摄像头的 FaceTime、Photo Booth 或 OBS。\n"
            "- 若仍无 iPhone，可继续使用 FaceTime 高清相机；OBS Virtual Camera 是虚拟摄像头。\n"
            "- 浏览器设备列表由浏览器权限控制，Python 无法直接读取或强制添加 iPhone。"
        )
    reset_key = "camera_browser_device_reset"
    if st.button(
        "刷新浏览器摄像头设备 / 重新请求摄像头权限",
        key="reset_browser_camera",
    ):
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        st.info(
            "已重建浏览器摄像头组件。请重新点击 Start、允许权限，再打开 Select Device。"
        )
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
    confidence_threshold: float,
    person_only: bool,
    acoustic_options: dict | None,
    tracking_mode: str = "person_yolo",
    object_tracker_config: dict | None = None,
) -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from src.browser_capture import FusionVideoProcessor

    if tracking_mode == "custom_object_template" and not object_tracker_config:
        st.error("请先录入目标图像并初始化指定物体追踪。")
        return
    if tracking_mode == "tennis_ball_color" and not object_tracker_config:
        st.error("请先测试并初始化网球标记追踪。")
        return

    st.info(
        "iPhone/Continuity Camera 用这条路径：浏览器选摄像头，本页选麦克风。"
        "设备列表由 macOS 和浏览器管理；iPhone 暂时不显示不会阻止其他摄像头采集。"
    )
    with st.expander("看不到 iPhone 摄像头？"):
        st.markdown(
            "- 先点 WebRTC 组件里的 `Start`，允许浏览器摄像头权限。\n"
            "- 再点 `Select Device`，从浏览器设备列表里选 iPhone。\n"
            "- 如果列表没有 iPhone，解锁并把 iPhone 靠近 Mac，确认 Wi-Fi、蓝牙和接力已开启。\n"
            "- 关闭可能占用摄像头的 FaceTime、Photo Booth 和 OBS，再刷新页面。\n"
            "- 确认 Safari/Chrome 已允许 localhost 使用摄像头。\n"
            "- 仍不可见时可使用 FaceTime 高清相机，或切换到 OpenCV 本机摄像头模式。"
        )
    reset_key = "fusion_browser_device_reset"
    if st.button(
        "刷新浏览器摄像头设备 / 重新请求摄像头权限",
        key="reset_browser_fusion",
    ):
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        st.info(
            "已重建浏览器摄像头组件。请重新点击 Start、允许权限，再打开 Select Device。"
        )
    component_generation = st.session_state.get(reset_key, 0)

    st.markdown("**切换路径**")
    st.markdown(
        "1. 点击下方 WebRTC 组件的 **Start**。\n"
        "2. 点击组件里的 **Select Device**，选择 iPhone 或其他浏览器可见摄像头。\n"
        "3. 麦克风在下面选择，推荐 `Wireless Mic Rx` 或系统默认麦克风。"
    )

    _audio_device_selector(key="browser_audio_device")
    local_audio_device_index = st.session_state.get("selected_audio_input_index")
    if local_audio_device_index is None:
        st.error("没有可用的 sounddevice 输入设备，无法开始浏览器摄像头 + 本机麦克风采集。")
        return
    local_audio_device_index = int(local_audio_device_index)
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
    if (
        session is None
        or session.finalized
        or session.status().get("tracking_mode") != tracking_mode
    ):
        session = SynchronizedCaptureSession(
            output_dir=OUTPUT_DIR,
            tracking_mode=tracking_mode,
            track_class=(
                "custom_object"
                if tracking_mode == "custom_object_template"
                else (
                    "tennis_ball_marker"
                    if tracking_mode == "tennis_ball_color"
                    else "person"
                )
            ),
        )
        st.session_state.fusion_session = session

    try:
        webrtc_options = {
            "key": _browser_component_key(
                "audio-visual-fusion", component_generation
            ),
            "mode": WebRtcMode.SENDRECV,
            "video_processor_factory": lambda: FusionVideoProcessor(
                session,
                confidence_threshold,
                person_only,
                tracking_mode=tracking_mode,
                object_tracker_config=object_tracker_config,
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
    tracking_slot = st.empty()

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
                st.session_state.last_audio_error = (
                    st.session_state.fusion_audio_start_error
                )
                st.warning(
                    "音频采集异常，请重新选择麦克风或重启采集。"
                    "浏览器视觉追踪仍可继续。错误："
                    + st.session_state.fusion_audio_start_error
                )
            else:
                try:
                    audio_worker = LocalAudioWorker(local_audio_device_index, session)
                    audio_worker.start(require_ready=True, timeout_sec=2.0)
                    st.session_state[audio_worker_key] = audio_worker
                    st.session_state.fusion_capture_active = True
                    st.success("音频流已就绪，开始视觉—音频同步采集。")
                except Exception as exc:
                    st.session_state.fusion_audio_start_error = str(exc)
                    st.session_state.last_audio_error = str(exc)
                    st.warning(
                        "音频采集异常，请重新选择麦克风或重启采集。"
                        "浏览器视觉追踪仍可继续。错误：" + str(exc)
                    )
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
        if tracking_mode in ("custom_object_template", "tennis_ball_color"):
            if status.get("tracking_status") == "lost":
                message = (
                    "网球目标暂时丢失；请调整 HSV 或将网球移回画面。"
                    if tracking_mode == "tennis_ball_color"
                    else "指定物体暂时丢失；请将物体移回 ROI 附近，必要时停止并重新初始化。"
                )
                tracking_slot.warning(message)
            else:
                tracking_label = (
                    "视觉模式：Tennis ball marker track｜"
                    if tracking_mode == "tennis_ball_color"
                    else "视觉模式：Custom object track｜"
                )
                tracking_slot.caption(
                    tracking_label
                    + f"lost={status.get('lost_frame_count', 0)}｜"
                    f"success={status.get('tracking_success_rate', 0.0):.1%}"
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
    devices_key = f"{key}_audio_devices"
    refresh_time_key = f"{key}_audio_refresh_time"
    choice_reason_key = f"{key}_audio_choice_reason"
    other_selector = (
        "browser_audio_device" if key == "local_audio_device" else "local_audio_device"
    )
    _stop_audio_level_monitor(other_selector)

    def preferred_device(devices: list[dict]) -> tuple[dict | None, str]:
        return find_preferred_input_device(
            devices,
            user_confirmed_index=st.session_state.get(
                "user_confirmed_wireless_mic_index"
            ),
            user_confirmed_name=st.session_state.get(
                "user_confirmed_wireless_mic_name"
            ),
        )

    def set_selection(
        index: int | None,
        devices: list[dict],
        reason: str,
        update_widget: bool = True,
    ) -> None:
        previous_index = st.session_state.get("selected_audio_input_index")
        if previous_index is not None and index is not None and previous_index != index:
            close_audio_resources()
        by_index = {item["index"]: item for item in devices}
        selected = by_index.get(index)
        if selected is None:
            st.session_state["selected_audio_input_index"] = None
            st.session_state["selected_audio_input_name"] = ""
            return
        if update_widget:
            st.session_state[key] = selected["index"]
        st.session_state["selected_audio_input_index"] = selected["index"]
        st.session_state["selected_audio_input_name"] = selected["name"]
        st.session_state[f"{key}_selected_audio_input_index"] = selected["index"]
        st.session_state[f"{key}_selected_audio_input_name"] = selected["name"]
        st.session_state[choice_reason_key] = reason

    refresh_col, prefer_col = st.columns(2)
    force_refresh = refresh_col.button("刷新麦克风列表", key=f"{key}_refresh")
    if force_refresh or devices_key not in st.session_state:
        try:
            devices = list_audio_input_devices(force_refresh=True)
        except AudioDeviceError as exc:
            st.error(str(exc))
            devices = []
        st.session_state[devices_key] = devices
        st.session_state[refresh_time_key] = datetime.now().strftime("%H:%M:%S")
        previous_index = st.session_state.get("selected_audio_input_index")
        available_indexes = {item["index"] for item in devices}
        preferred, reason = preferred_device(devices)
        confirmed_match = preferred is not None and (
            "用户确认" in reason or "重新匹配" in reason
        )
        if confirmed_match:
            set_selection(preferred["index"], devices, reason)
        elif previous_index in available_indexes:
            set_selection(previous_index, devices, "刷新后保留原选择 index")
        else:
            set_selection(preferred["index"] if preferred else None, devices, reason)

    devices = st.session_state.get(devices_key, [])
    if prefer_col.button("优先选择无线麦接收器", key=f"{key}_prefer_wireless"):
        preferred, reason = preferred_device(devices)
        set_selection(preferred["index"] if preferred else None, devices, reason)

    st.caption(
        "刷新会重新调用 sounddevice.query_devices()。"
        "若仍未出现 Wireless Mic Rx，请重新插拔 USB 接收器或重启 Streamlit。"
    )
    if st.session_state.get(refresh_time_key):
        st.caption(f"最近刷新时间：{st.session_state[refresh_time_key]}")
    if not devices:
        st.session_state["selected_audio_input_index"] = None
        st.session_state["selected_audio_input_name"] = ""
        st.error("没有枚举到可用输入设备，无法开始同步采集。")
        return None

    available_indexes = [item["index"] for item in devices]
    current_index = st.session_state.get("selected_audio_input_index")
    if current_index not in available_indexes:
        preferred, reason = preferred_device(devices)
        set_selection(preferred["index"] if preferred else None, devices, reason)
    labels = {item["index"]: item["display_name"] for item in devices}
    default_index = get_default_input_device_index()
    selected_index = st.selectbox(
        "麦克风输入设备",
        options=available_indexes,
        format_func=lambda value: (
            "默认麦克风：" + labels[value]
            if value == default_index
            else labels[value]
        ),
        key=key,
    )
    selected = next(item for item in devices if item["index"] == selected_index)
    set_selection(
        selected_index,
        devices,
        "用户在下拉框选择",
        update_widget=False,
    )
    st.caption(
        "正式采集固定使用 sounddevice input index："
        f"{selected['index']}（{selected['name']}）。"
    )
    if st.session_state.get(choice_reason_key):
        st.info("当前选择原因：" + st.session_state[choice_reason_key])
    selected_classification = classify_input_device(selected)
    if not any(
        classify_input_device(item)["is_exact_wireless_mic_rx"] for item in devices
    ):
        if any(classify_input_device(item)["is_usb_audio_codec"] for item in devices):
            st.warning(
                "未枚举到精确名称 Wireless Mic Rx，但发现 USB audio CODEC。"
                "它可能就是同一无线麦接收器的 PortAudio 名称，可直接测试输入电平。"
            )
        else:
            st.warning("未枚举到 Wireless Mic Rx。请点击刷新，或查看下方“麦克风诊断”建议。")
    if selected_classification["is_possible_wireless_receiver"]:
        st.caption("当前设备类型：" + selected_classification["friendly_type"])
    _render_audio_input_level(selected_index, key=f"{key}_level")
    return int(st.session_state["selected_audio_input_index"])


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
    errors = close_audio_resources()
    for selector_key in ("local_audio_device", "browser_audio_device"):
        st.session_state.pop(f"{selector_key}_audio_devices", None)
        st.session_state.pop(f"{selector_key}_last_audio_test", None)
    st.session_state["last_audio_error"] = "；".join(errors)
    st.session_state["fusion_audio_start_error"] = ""


def close_audio_resources() -> list[str]:
    """Best-effort close of every session-owned PortAudio stream."""
    errors: list[str] = []
    for key in ("local_audio_device", "browser_audio_device"):
        try:
            _stop_audio_level_monitor(key)
        except Exception as exc:
            errors.append(str(exc))
    browser_worker = st.session_state.get("browser_fusion_audio_worker")
    if browser_worker is not None:
        try:
            browser_worker.close()
        except Exception as exc:
            errors.append(str(exc))
        st.session_state["browser_fusion_audio_worker"] = None
    local_worker = st.session_state.get("local_fusion_worker")
    if local_worker is not None:
        try:
            local_worker.audio_capture.close()
        except Exception as exc:
            errors.append(str(exc))
    session = st.session_state.get("fusion_session")
    if session is not None:
        try:
            session.audio_recorder.close()
        except Exception as exc:
            errors.append(str(exc))
    return errors


def _microphone_diagnostics(
    selector_key: str,
    selected_index: int | None,
    capture_status: dict | None = None,
) -> None:
    with st.expander("麦克风诊断", expanded=False):
        test_key = f"{selector_key}_last_audio_test"
        test_result = st.session_state.get(test_key)
        level_evidence: list[bool] = []
        if test_result and test_result.get("device_index") == selected_index:
            level_evidence.append(
                bool(test_result.get("ok"))
                and float(test_result.get("peak", 0.0) or 0.0) > 1e-6
            )
        monitor = st.session_state.get(f"{selector_key}_level_monitor")
        if monitor is not None and monitor.audio_device_index == selected_index:
            monitor_level = monitor.status()
            level_evidence.append(float(monitor_level.get("peak", 0.0) or 0.0) > 1e-6)
        if capture_status and capture_status.get("device_index") == selected_index:
            level_evidence.append(
                bool(capture_status.get("receiving"))
                and float(capture_status.get("last_rms", 0.0) or 0.0) > 1e-7
            )
        selected_has_level = any(level_evidence) if level_evidence else None
        diagnostics = diagnose_audio_devices(
            selected_index=selected_index,
            selected_device_has_level=selected_has_level,
            user_confirmed_index=st.session_state.get(
                "user_confirmed_wireless_mic_index"
            ),
            user_confirmed_name=st.session_state.get(
                "user_confirmed_wireless_mic_name"
            ),
        )
        if not diagnostics["ok"]:
            st.error("sounddevice 诊断失败：" + diagnostics["error"])
            return
        devices = diagnostics["devices"]
        selected_device = diagnostics["selected_device"]
        selected_display = selected_device["display_name"] if selected_device else "--"
        selected_type = diagnostics["selected_device_classification"]["friendly_type"]
        st.write(f"当前选择设备：`{selected_display}`")
        st.write(f"当前选择设备类型：`{selected_type}`")
        st.write(f"当前选择设备可用：`{diagnostics['selected_device_available']}`")
        level_text = "尚无电平数据" if selected_has_level is None else str(selected_has_level)
        st.write(f"当前选择设备有输入电平：`{level_text}`")
        st.write(f"当前系统默认 input index：`{diagnostics['default_input_index']}`")
        st.write(f"sounddevice 版本：`{diagnostics['sounddevice_version']}`")
        st.write(f"sounddevice 默认设备：`{diagnostics['default_device']}`")
        st.write(
            "最近一次音频错误：`"
            + (st.session_state.get("last_audio_error") or "--")
            + "`"
        )
        st.write(
            "是否发现精确名称 Wireless Mic Rx："
            f"`{diagnostics['found_exact_wireless_mic_rx']}`"
        )
        st.write(
            "是否发现可能的无线麦接收器："
            f"`{diagnostics['found_possible_wireless_receiver']}`"
        )
        st.write(
            f"是否发现 USB audio CODEC：`{diagnostics['found_usb_audio_codec']}`"
        )
        st.write(
            "推荐设备：`"
            + (
                diagnostics["recommended_device"]["display_name"]
                if diagnostics["recommended_device"]
                else "--"
            )
            + "`"
        )
        st.caption("推荐原因：" + diagnostics["recommendation_reason"])
        if (
            not diagnostics["found_exact_wireless_mic_rx"]
            and diagnostics["found_usb_audio_codec"]
        ):
            st.warning(diagnostics["recommendation_message"])
        else:
            st.info(diagnostics["recommendation_message"])
        if st.button(
            "将当前设备作为我的 Wireless Mic Rx 使用",
            key=f"{selector_key}_confirm_wireless",
            disabled=selected_device is None,
        ):
            st.session_state["user_confirmed_wireless_mic_index"] = selected_index
            st.session_state["user_confirmed_wireless_mic_name"] = (
                selected_device["name"] if selected_device else ""
            )
            st.success(
                f"已记住 {selected_device['name']} | index={selected_index}。"
                "后续刷新和推荐会优先选择它。"
            )
        confirmed_index = st.session_state.get("user_confirmed_wireless_mic_index")
        confirmed_name = st.session_state.get("user_confirmed_wireless_mic_name", "")
        if confirmed_index is not None:
            st.caption(
                f"我的 Wireless Mic Rx：{confirmed_name} | index={confirmed_index}"
            )
        refresh_time = st.session_state.get(f"{selector_key}_audio_refresh_time")
        if refresh_time:
            st.caption(f"页面最近刷新时间：{refresh_time}")
        if devices:
            st.caption("当前可见输入设备")
            for item in devices:
                st.text(item["display_name"] + f" | hostapi={item['hostapi']}")
        else:
            st.warning("sounddevice 当前没有枚举到输入设备。")
        for suggestion in diagnostics["suggestions"]:
            st.info(suggestion)

        button_col, reset_col = st.columns(2)
        if button_col.button("测试当前麦克风 1 秒", key=f"{selector_key}_test_audio"):
            _stop_audio_level_monitor(selector_key)
            result = test_audio_input_device(selected_index, duration=1.0)
            st.session_state[test_key] = result
        if reset_col.button("重置音频采集状态", key=f"{selector_key}_reset_audio"):
            _reset_audio_capture_state()
            st.success("已关闭旧音频流、清理状态；页面重新运行后会重新枚举设备。")

        test_result = st.session_state.get(test_key)
        if test_result:
            if test_result.get("ok"):
                st.success(
                    "最近测试成功："
                    f"{test_result['device_name']} | index={test_result['device_index']} | "
                    f"{test_result['sample_rate']} Hz | "
                    f"RMS {test_result['rms']:.5f} | {test_result['db']:.1f} dBFS"
                )
                selected_classification = classify_input_device(
                    {"name": test_result.get("device_name", "")}
                )
                if (
                    selected_classification["is_usb_audio_codec"]
                    and float(test_result.get("peak", 0.0) or 0.0) > 1e-6
                ):
                    st.info(
                        "当前 USB audio CODEC 可能就是 Wireless Mic Rx 的 "
                        "PortAudio 名称。"
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
    backend = (
        f"{status.get('object_tracker_type') or 'tracker'}"
        if status.get("tracking_mode") in ("custom_object_template", "tennis_ball_color")
        else status["backend"]
    )
    st.caption(
        f"检测后端：{backend}｜视觉样本 {status['visual_samples']} 条｜"
        f"音频样本 {status['audio_samples']} 条｜已采集 {status['elapsed_sec']:.1f} 秒"
    )
    if status.get("tracking_mode") in ("custom_object_template", "tennis_ball_color"):
        if status.get("object_tracking_status") == "lost":
            st.warning(
                "网球目标暂时丢失。"
                if status.get("tracking_mode") == "tennis_ball_color"
                else "目标暂时丢失。"
            )
        if status.get("consecutive_lost_frames", 0) >= 30:
            st.error(
                "网球已连续丢失 30 帧以上，建议停止采集并调整 HSV 或重新初始化。"
                if status.get("tracking_mode") == "tennis_ball_color"
                else "指定物体已连续丢失 30 帧以上，建议停止采集并重新初始化目标。"
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
    confidence_threshold: float,
    person_only: bool,
    acoustic_options: dict | None,
    tracking_mode: str = "person_yolo",
    object_tracker_config: dict | None = None,
) -> None:
    st.info("本机固定实验用这条路径：下方选择 OpenCV 摄像头和 sounddevice 麦克风。")
    camera_index = _camera_index_controls("fusion")
    _audio_device_selector(key="local_audio_device")
    audio_device_index = st.session_state.get("selected_audio_input_index")
    if audio_device_index is not None:
        audio_device_index = int(audio_device_index)
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
        "开始同步采集",
        type="primary",
        disabled=(
            bool(worker and worker.running)
            or audio_device_index is None
            or (
                tracking_mode in ("custom_object_template", "tennis_ball_color")
                and not object_tracker_config
            )
        ),
    ):
        if tracking_mode == "custom_object_template" and not object_tracker_config:
            st.error("请先录入目标图像并初始化指定物体追踪。")
            return
        if tracking_mode == "tennis_ball_color" and not object_tracker_config:
            st.error("请先测试并初始化网球标记追踪。")
            return
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
                    st.session_state.last_audio_error = str(audio_test.get("error", ""))
                    st.warning(
                        "麦克风预检失败，将尝试有限设备回退；即使音频不可用，"
                        "视觉追踪仍会启动。错误：" + str(audio_test.get("error", ""))
                    )
                if worker is not None:
                    worker.stop()
                worker = LocalFusionWorker(
                    camera_index=camera_index,
                    audio_device_index=audio_device_index,
                    output_dir=OUTPUT_DIR,
                    confidence_threshold=confidence_threshold,
                    person_only=person_only,
                    tracking_mode=tracking_mode,
                    object_tracker_config=object_tracker_config,
                )
                worker.start()
                st.session_state[worker_key] = worker
                if worker.audio_error:
                    st.session_state.last_audio_error = worker.audio_error
                    st.warning(
                        "音频采集异常，请重新选择麦克风或重启采集；"
                        "视觉追踪已继续运行。错误：" + worker.audio_error
                    )
                else:
                    st.success("音频流已就绪，开始视觉—音频同步采集。")
            except Exception as exc:
                st.session_state.last_audio_error = str(exc)
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
    with st.expander("声学轨迹导出设置", expanded=False):
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


def _store_object_template(frame, signature: str) -> None:
    st.session_state.custom_object_template_frame = frame.copy()
    st.session_state.custom_object_template_signature = signature
    st.session_state.custom_object_tracker_ready = False
    st.session_state.custom_object_tracker_config = None
    height, width = frame.shape[:2]
    st.session_state.custom_roi_x = max(0, width // 4)
    st.session_state.custom_roi_y = max(0, height // 4)
    st.session_state.custom_roi_width = max(1, width // 2)
    st.session_state.custom_roi_height = max(1, height // 2)


def _store_tennis_template(frame, signature: str) -> None:
    st.session_state.tennis_template_frame = frame.copy()
    st.session_state.tennis_template_signature = signature
    st.session_state.tennis_tracker_ready = False
    st.session_state.tennis_tracker_config = None
    height, width = frame.shape[:2]
    st.session_state.tennis_roi_x = max(0, width // 4)
    st.session_state.tennis_roi_y = max(0, height // 4)
    st.session_state.tennis_roi_width = max(1, width // 2)
    st.session_state.tennis_roi_height = max(1, height // 2)


def _tennis_tracking_advanced_controls() -> tuple[str, dict | None]:
    source = st.radio(
        "网球颜色参数来源",
        [
            "使用默认网球颜色阈值",
            "从当前 OpenCV 摄像头画面截帧并估计",
            "上传包含网球的图片并估计",
        ],
        horizontal=True,
        key="tennis_template_source",
    )
    camera_index = int(st.session_state.get("fusion_camera_index", 0))
    if source.startswith("从当前"):
        st.caption(f"将从 OpenCV Camera {camera_index} 截取一帧用于 ROI 设置。")
        if st.button("截取网球当前帧", key="capture_tennis_template"):
            try:
                frame = read_camera_preview(camera_index)
                _store_tennis_template(frame, f"camera-{camera_index}-{time.time_ns()}")
                st.success("已截取当前帧，请把 ROI 调整到网球区域。")
            except Exception as exc:
                st.error(f"网球当前帧截取失败：{exc}")
    elif source.startswith("上传"):
        uploaded = st.file_uploader(
            "上传包含网球的图片（建议与采集机位相同）",
            type=["jpg", "jpeg", "png", "bmp"],
            key="tennis_template_upload",
        )
        if uploaded is not None:
            signature = f"{uploaded.name}-{uploaded.size}"
            if st.session_state.get("tennis_template_signature") != signature:
                data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("无法读取网球图片。")
                else:
                    _store_tennis_template(frame, signature)

    template = st.session_state.get("tennis_template_frame")
    if template is not None:
        frame_height, frame_width = template.shape[:2]
        for key, upper in (
            ("tennis_roi_x", max(0, frame_width - 1)),
            ("tennis_roi_y", max(0, frame_height - 1)),
            ("tennis_roi_width", frame_width),
            ("tennis_roi_height", frame_height),
        ):
            current = int(
                st.session_state.get(key, 0 if key.endswith(("_x", "_y")) else 1)
            )
            st.session_state[key] = min(
                max(current, 0 if key.endswith(("_x", "_y")) else 1), upper
            )
        st.markdown("**网球 ROI 设置（图片像素坐标）**")
        columns = st.columns(4)
        roi_x = columns[0].number_input(
            "ROI x", min_value=0, max_value=max(0, frame_width - 1), key="tennis_roi_x"
        )
        roi_y = columns[1].number_input(
            "ROI y", min_value=0, max_value=max(0, frame_height - 1), key="tennis_roi_y"
        )
        max_width = max(1, frame_width - int(roi_x))
        max_height = max(1, frame_height - int(roi_y))
        if st.session_state.tennis_roi_width > max_width:
            st.session_state.tennis_roi_width = max_width
        if st.session_state.tennis_roi_height > max_height:
            st.session_state.tennis_roi_height = max_height
        roi_width = columns[2].number_input(
            "ROI width", min_value=1, max_value=max_width, key="tennis_roi_width"
        )
        roi_height = columns[3].number_input(
            "ROI height", min_value=1, max_value=max_height, key="tennis_roi_height"
        )
        bbox = (int(roi_x), int(roi_y), int(roi_width), int(roi_height))
        preview = template.copy()
        cv2.rectangle(
            preview,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (0, 255, 0),
            3,
        )
        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption=f"网球 ROI 预览 | {frame_width}x{frame_height}",
            width=640,
        )
    else:
        bbox = None
        st.caption(
            "默认阈值可以直接用于实时采集；若要自动估计颜色，请先截帧或上传图片。"
        )

    hsv_defaults = {
        "tennis_h_range": (DEFAULT_HSV_LOWER[0], DEFAULT_HSV_UPPER[0]),
        "tennis_s_range": (DEFAULT_HSV_LOWER[1], DEFAULT_HSV_UPPER[1]),
        "tennis_v_range": (DEFAULT_HSV_LOWER[2], DEFAULT_HSV_UPPER[2]),
    }
    for key, value in hsv_defaults.items():
        st.session_state.setdefault(key, value)

    if template is not None and bbox is not None and st.button(
        "从 ROI 自动估计 HSV", key="estimate_tennis_hsv"
    ):
        try:
            lower, upper = estimate_hsv_range_from_roi(template, bbox)
            st.session_state.tennis_h_range = (lower[0], upper[0])
            st.session_state.tennis_s_range = (lower[1], upper[1])
            st.session_state.tennis_v_range = (lower[2], upper[2])
            st.session_state.tennis_hsv_estimated = (lower, upper)
            st.success(f"已估计 HSV：lower={lower}，upper={upper}")
        except Exception as exc:
            st.error(f"HSV 估计失败：{exc}")

    st.markdown("**HSV 阈值（OpenCV H 范围为 0–179）**")
    hsv_columns = st.columns(3)
    h_range = hsv_columns[0].slider("H min / max", 0, 179, key="tennis_h_range")
    s_range = hsv_columns[1].slider("S min / max", 0, 255, key="tennis_s_range")
    v_range = hsv_columns[2].slider("V min / max", 0, 255, key="tennis_v_range")
    min_area = st.number_input(
        "网球最小面积", min_value=1.0, max_value=1_000_000.0, value=80.0, step=10.0,
        key="tennis_min_area",
    )
    max_area = st.number_input(
        "网球最大面积", min_value=2.0, max_value=2_000_000.0, value=50_000.0, step=100.0,
        key="tennis_max_area",
    )
    min_circularity = st.slider(
        "最小圆度", 0.05, 1.0, 0.45, 0.05, key="tennis_min_circularity"
    )
    hsv_lower = (int(h_range[0]), int(s_range[0]), int(v_range[0]))
    hsv_upper = (int(h_range[1]), int(s_range[1]), int(v_range[1]))
    params = {
        "hsv_lower": hsv_lower,
        "hsv_upper": hsv_upper,
        "min_area": float(min_area),
        "max_area": float(max_area),
        "min_circularity": float(min_circularity),
        "smoothing": 0.35,
        "max_lost_frames": 30,
    }

    if template is not None:
        mask = make_tennis_mask(template, hsv_lower, hsv_upper)
        st.image(mask, caption="当前 HSV mask 预览（白色为候选区域）", clamp=True, width=640)
    else:
        st.caption("mask 预览需要先截取或上传一张包含网球的图片。")

    if st.button("测试网球识别", key="test_tennis_tracker"):
        try:
            test_frame = template
            if test_frame is None:
                test_frame = read_camera_preview(camera_index)
                _store_tennis_template(test_frame, f"test-camera-{time.time_ns()}")
            tracker = TennisBallTracker(**params)
            result = tracker.update(test_frame)
            st.session_state.tennis_last_test = {
                "result": result,
                "params": params.copy(),
            }
        except Exception as exc:
            st.session_state.tennis_last_test = {
                "result": {"ok": False, "status": "lost", "error": str(exc)},
                "params": params.copy(),
            }
    last_test = st.session_state.get("tennis_last_test")
    if last_test and last_test.get("params") == params:
        result = last_test["result"]
        if result.get("ok"):
            st.success(
                "网球识别成功："
                f"center=({result['center_x']:.1f}, {result['center_y']:.1f})，"
                f"radius={result['marker_radius']:.1f}，area={result['marker_area']:.1f}，"
                f"circularity={result['marker_circularity']:.2f}，status=tracking"
            )
        else:
            st.warning("当前帧没有识别到网球，请调整 HSV、面积或圆度阈值后重试。")
    elif last_test:
        st.caption("HSV 或筛选参数已变化，请重新测试网球识别。")

    if st.button("初始化网球追踪", type="primary", key="initialize_tennis_tracker"):
        if not last_test or last_test.get("params") != params or not last_test.get("result", {}).get("ok"):
            st.error("请先测试并成功识别网球标记。")
        else:
            st.session_state.tennis_tracker_config = params.copy()
            st.session_state.tennis_tracker_ready = True
            st.success(
                "网球标记追踪已就绪：tracking_mode=tennis_ball_color，"
                "track_id=1，状态=ready。"
            )

    config = st.session_state.get("tennis_tracker_config")
    if config and config != params:
        st.session_state.tennis_tracker_ready = False
        st.session_state.tennis_tracker_config = None
        config = None
        st.warning("HSV 或筛选参数已变化，请重新测试并初始化网球追踪。")
    if st.session_state.get("tennis_tracker_ready") and config:
        st.caption("当前网球追踪配置已就绪，正式 START 将使用这些 HSV 和筛选参数。")
        return "tennis_ball_color", dict(config)
    st.error("请先测试并初始化网球标记追踪，之后才能 START。")
    return "tennis_ball_color", None


def _tennis_tracking_controls() -> tuple[str, dict | None]:
    st.info(
        "当前使用网球颜色标记追踪。若识别稳定，通常无需展开高级参数。"
        "该模式使用 HSV 颜色分割，不受 YOLO person 过滤影响。"
    )
    with st.expander("网球颜色与识别高级参数", expanded=False):
        return _tennis_tracking_advanced_controls()


def _object_tracking_controls() -> tuple[str, dict | None]:
    target_label = st.radio(
        "视觉追踪目标类型",
        [
            YOLO_TRACKING_LABEL,
            TENNIS_TRACKING_LABEL,
            CUSTOM_OBJECT_TRACKING_LABEL,
        ],
        index=None,
        horizontal=True,
        key="fusion_tracking_target_type",
    )
    if target_label.startswith("YOLO"):
        return "person_yolo", None
    if target_label.startswith("Tennis"):
        return _tennis_tracking_controls()

    st.warning(
        "指定物体追踪适合颜色、形状或纹理明显的物体。遮挡、快速旋转、"
        "尺度或光照变化可能导致丢失。麦克风等小物体建议贴明显颜色标记；"
        "更稳定的后续方案可考虑 ArUco marker 或 Siamese tracker。"
    )
    source = st.radio(
        "目标录入方式",
        ["使用当前 OpenCV 摄像头画面截帧", "上传目标模板图片"],
        horizontal=True,
        key="custom_object_template_source",
    )
    if source.startswith("使用当前"):
        camera_index = int(st.session_state.get("fusion_camera_index", 0))
        st.caption(
            f"将从 OpenCV Camera {camera_index} 截帧。浏览器/iPhone 模式建议上传同一机位的完整首帧。"
        )
        if st.button("截取当前帧作为模板", key="capture_custom_object_template"):
            try:
                frame = read_camera_preview(camera_index)
                _store_object_template(frame, f"camera-{camera_index}-{time.time_ns()}")
                st.success("已截取当前摄像头画面。请设置 ROI。")
            except Exception as exc:
                st.error(f"目标模板截帧失败：{exc}")
    else:
        uploaded = st.file_uploader(
            "上传目标模板图片（建议使用与采集相同机位的完整画面）",
            type=["jpg", "jpeg", "png", "bmp"],
            key="custom_object_template_upload",
        )
        if uploaded is not None:
            signature = f"{uploaded.name}-{uploaded.size}"
            if st.session_state.get("custom_object_template_signature") != signature:
                data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("无法读取目标模板图片。")
                else:
                    _store_object_template(frame, signature)

    template = st.session_state.get("custom_object_template_frame")
    if template is None:
        st.info("请先截取当前帧或上传目标模板图片。")
        return "custom_object_template", None

    frame_height, frame_width = template.shape[:2]
    for key, upper in (
        ("custom_roi_x", max(0, frame_width - 1)),
        ("custom_roi_y", max(0, frame_height - 1)),
        ("custom_roi_width", frame_width),
        ("custom_roi_height", frame_height),
    ):
        current = int(st.session_state.get(key, 0 if key.endswith(("_x", "_y")) else 1))
        st.session_state[key] = min(max(current, 0 if key.endswith(("_x", "_y")) else 1), upper)

    st.markdown("**ROI 设置（模板图像像素坐标）**")
    columns = st.columns(4)
    roi_x = columns[0].number_input(
        "ROI x", min_value=0, max_value=max(0, frame_width - 1), key="custom_roi_x"
    )
    roi_y = columns[1].number_input(
        "ROI y", min_value=0, max_value=max(0, frame_height - 1), key="custom_roi_y"
    )
    max_width = max(1, frame_width - int(roi_x))
    max_height = max(1, frame_height - int(roi_y))
    if st.session_state.custom_roi_width > max_width:
        st.session_state.custom_roi_width = max_width
    if st.session_state.custom_roi_height > max_height:
        st.session_state.custom_roi_height = max_height
    roi_width = columns[2].number_input(
        "ROI width", min_value=1, max_value=max_width, key="custom_roi_width"
    )
    roi_height = columns[3].number_input(
        "ROI height", min_value=1, max_value=max_height, key="custom_roi_height"
    )
    bbox = (int(roi_x), int(roi_y), int(roi_width), int(roi_height))
    preview = template.copy()
    cv2.rectangle(
        preview,
        (bbox[0], bbox[1]),
        (bbox[0] + bbox[2], bbox[1] + bbox[3]),
        (0, 255, 0),
        3,
    )
    st.image(
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        caption=f"目标模板与 ROI | {frame_width}x{frame_height}",
        width=640,
    )
    requested_tracker = st.selectbox(
        "OpenCV tracker（不可用时自动回退）",
        ["CSRT", "KCF", "MIL"],
        key="custom_object_tracker_type",
    )
    if st.button(
        "初始化指定物体追踪",
        type="primary",
        key="initialize_custom_object_tracker",
    ):
        try:
            validate_bbox(template, bbox)
            tracker = ObjectTemplateTracker(requested_tracker)
            initialized = tracker.initialize(template, bbox)
        except Exception as exc:
            initialized = False
            tracker = None
            st.error(f"指定物体追踪初始化失败：{exc}")
        if initialized and tracker is not None:
            config = {
                "tracker_type": requested_tracker,
                "actual_tracker_type": tracker.actual_tracker_type,
                "bbox": bbox,
                "template_size": (frame_width, frame_height),
                "template_signature": st.session_state.get(
                    "custom_object_template_signature", ""
                ),
            }
            st.session_state.custom_object_tracker_config = config
            st.session_state.custom_object_tracker_ready = True
            st.success(
                "指定物体追踪已就绪："
                f"tracker={tracker.actual_tracker_type}，bbox={bbox}，"
                "object track_id=1，状态=ready。"
            )
        elif tracker is not None:
            st.session_state.custom_object_tracker_ready = False
            st.session_state.custom_object_tracker_config = None
            st.error("指定物体追踪初始化失败：" + tracker.last_error)

    config = st.session_state.get("custom_object_tracker_config")
    current_signature = st.session_state.get("custom_object_template_signature", "")
    if config and (
        tuple(config.get("bbox", ())) != bbox
        or config.get("template_signature") != current_signature
        or config.get("tracker_type") != requested_tracker
    ):
        st.session_state.custom_object_tracker_ready = False
        st.warning("模板或 ROI 已变化，请重新点击“初始化指定物体追踪”。")
        config = None
    if st.session_state.get("custom_object_tracker_ready") and config:
        st.success(
            "当前指定物体配置已就绪："
            f"{config['actual_tracker_type']} | bbox={tuple(config['bbox'])} | track_id=1"
        )
        return "custom_object_template", dict(config)
    st.error("请先录入目标图像并初始化指定物体追踪。")
    return "custom_object_template", None


def fusion_capture_mode(confidence_threshold: float, person_only: bool) -> None:
    st.subheader("视觉—音频同步采集")
    acoustic_options = _acoustic_export_controls()
    tracking_mode, object_tracker_config = _object_tracking_controls()
    source = st.radio(
        "采集方式",
        [
            "浏览器摄像头 / 麦克风（推荐用于 iPhone Continuity Camera）",
            "OpenCV 摄像头 + sounddevice 麦克风（推荐用于本地稳定实验）",
        ],
        horizontal=True,
        key="fusion_capture_source",
    )
    previous_source = st.session_state.get("active_fusion_capture_source")
    if previous_source is not None and previous_source != source:
        close_audio_resources()
        local_worker = st.session_state.get("local_fusion_worker")
        if local_worker is not None and getattr(local_worker, "running", False):
            local_worker.stop()
    st.session_state["active_fusion_capture_source"] = source
    if source.startswith("浏览器"):
        _browser_fusion_capture(
            confidence_threshold,
            person_only,
            acoustic_options,
            tracking_mode,
            object_tracker_config,
        )
    else:
        _local_fusion_capture(
            confidence_threshold,
            person_only,
            acoustic_options,
            tracking_mode,
            object_tracker_config,
        )


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
        st.markdown(
            "**摄像头诊断建议**\n\n"
            "- iPhone Continuity Camera 可能需要解锁 iPhone、靠近 Mac，并开启蓝牙、Wi-Fi 和接力。\n"
            "- Select Device 中没有 iPhone 时，刷新页面并关闭 FaceTime、Photo Booth 或 OBS。\n"
            "- FaceTime 高清相机可作为稳定备用。\n"
            "- OBS Virtual Camera 是虚拟摄像头，不是 iPhone 摄像头。\n"
            "- iPhone 不可见只是诊断提示，不代表浏览器摄像头采集失败。"
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
    if mode == "视觉—音频同步采集":
        st.session_state.setdefault(
            "fusion_tracking_target_type", TENNIS_TRACKING_LABEL
        )
        if st.session_state.fusion_tracking_target_type == "YOLO person（人形追踪，默认）":
            st.session_state.fusion_tracking_target_type = YOLO_TRACKING_LABEL
    fusion_tracking_label = st.session_state.get(
        "fusion_tracking_target_type", TENNIS_TRACKING_LABEL
    )
    st.sidebar.subheader("检测设置")
    confidence_threshold = st.sidebar.slider(
        "conf 阈值", min_value=0.1, max_value=0.9, value=0.25, step=0.05
    )
    if (
        mode == "视觉—音频同步采集"
        and fusion_tracking_label != YOLO_TRACKING_LABEL
    ):
        person_only = True
        if fusion_tracking_label == TENNIS_TRACKING_LABEL:
            st.sidebar.caption(
                "当前视觉追踪模式为 Tennis ball marker：使用颜色分割追踪网球，"
                "不使用 YOLO person 过滤。"
            )
        else:
            st.sidebar.caption(
                "当前视觉追踪模式为指定物体模板追踪，不使用 YOLO person 过滤。"
            )
    else:
        person_only = st.sidebar.checkbox(
            "只检测 person", value=True, key="yolo_person_only"
        )
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
