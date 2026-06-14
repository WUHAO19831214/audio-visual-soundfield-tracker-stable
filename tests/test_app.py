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
    fusion.radio[0].set_value(
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
