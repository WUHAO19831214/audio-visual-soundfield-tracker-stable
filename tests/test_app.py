from streamlit.testing.v1 import AppTest


def test_fusion_page_renders() -> None:
    app = AppTest.from_file("app.py")
    app.run(timeout=30)
    assert not app.exception

    app.radio[0].set_value("视觉—音频同步采集")
    app.run(timeout=30)

    assert not app.exception
    assert any(
        "视觉—音频同步采集" in heading.value for heading in app.subheader
    )


def test_browser_capture_pages_fail_gracefully_in_test_runtime() -> None:
    camera = AppTest.from_file("app.py").run(timeout=30)
    camera.radio[0].set_value("摄像头实时计数").run(timeout=30)
    camera.radio[0].set_value(
        "浏览器摄像头（推荐用于 iPhone Continuity Camera）"
    ).run(timeout=30)
    assert not camera.exception

    fusion = AppTest.from_file("app.py").run(timeout=30)
    fusion.radio[0].set_value("视觉—音频同步采集").run(timeout=30)
    capture_source = next(
        radio for radio in fusion.radio if radio.key == "fusion_capture_source"
    )
    capture_source.set_value(
        "浏览器摄像头 / 麦克风（推荐用于 iPhone Continuity Camera）"
    ).run(timeout=30)
    assert not fusion.exception


def test_browser_constraints_never_include_empty_exact_device_id() -> None:
    source = open("app.py", encoding="utf-8").read()

    assert 'deviceId": {"exact": ""}' not in source
    assert '_browser_component_key("camera-counting"' in source
    assert '"audio-visual-fusion", component_generation' in source


def test_device_diagnostics_area_renders() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)

    assert not app.exception
    assert any(item.label == "设备诊断" for item in app.expander)


def test_custom_object_tracking_setup_renders_without_template() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    app.radio[0].set_value("视觉—音频同步采集").run(timeout=30)
    target_type = next(
        radio for radio in app.radio if radio.key == "fusion_tracking_target_type"
    )

    target_type.set_value("指定物体（模板追踪，实验性）").run(timeout=30)

    assert not app.exception
    assert any(
        button.label == "截取当前帧作为模板" for button in app.button
    )


def test_tennis_tracking_setup_renders_without_frame() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    app.radio[0].set_value("视觉—音频同步采集").run(timeout=30)
    target_type = next(
        radio for radio in app.radio if radio.key == "fusion_tracking_target_type"
    )

    assert not app.exception
    assert target_type.value == "Tennis ball marker（网球标记追踪，推荐）"
    assert all(checkbox.label != "只检测 person" for checkbox in app.checkbox)
    assert any(
        "不使用 YOLO person 过滤" in caption.value for caption in app.caption
    )
    assert any(button.label == "测试网球识别" for button in app.button)
    assert any(button.label == "初始化网球追踪" for button in app.button)


def test_yolo_tracking_keeps_person_only_filter() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    app.radio[0].set_value("视觉—音频同步采集").run(timeout=30)
    target_type = next(
        radio for radio in app.radio if radio.key == "fusion_tracking_target_type"
    )

    target_type.set_value("YOLO person（人形追踪）").run(timeout=30)

    person_only = next(
        checkbox for checkbox in app.checkbox if checkbox.label == "只检测 person"
    )
    assert not app.exception
    assert person_only.value is True


def test_advanced_capture_settings_are_collapsed_by_default() -> None:
    source = open("app.py", encoding="utf-8").read()

    assert 'st.expander("声学轨迹导出设置", expanded=False)' in source
    assert 'st.expander("网球颜色与识别高级参数", expanded=False)' in source
