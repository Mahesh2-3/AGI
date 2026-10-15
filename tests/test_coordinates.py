from vision.coordinates import CoordinateTransformer


def test_normalized_coordinates_scaling():
    transformer = CoordinateTransformer(orig_w=1920, orig_h=1080, model_w=1280, model_h=720)
    # Center normalized [0.5, 0.5]
    px, py = transformer.to_screen_pixels(0.5, 0.5)
    assert px == 960
    assert py == 540


def test_model_pixel_scaling():
    # If model returns pixels in its 1280x720 view
    transformer = CoordinateTransformer(orig_w=1920, orig_h=1080, model_w=1280, model_h=720)
    # Model center is (640, 360) -> should map to screen (960, 540)
    px, py = transformer.to_screen_pixels(640, 360)
    assert px == 960
    assert py == 540


def test_bbox_ordering_and_clamping():
    transformer = CoordinateTransformer(orig_w=1920, orig_h=1080, model_w=1280, model_h=720)
    # Inverted and out-of-bounds normalized box
    res = transformer.transform_bbox([0.8, 0.9, 0.2, 0.1])
    assert res is not None
    # Must be ordered xmin <= xmax, ymin <= ymax
    assert res["bbox"][0] <= res["bbox"][2]
    assert res["bbox"][1] <= res["bbox"][3]
    assert res["bbox_pixels"][0] == int(round(0.2 * 1920))
    assert res["bbox_pixels"][1] == int(round(0.1 * 1080))


def test_click_point_inside_vs_outside_bbox():
    transformer = CoordinateTransformer(orig_w=1920, orig_h=1080, model_w=1920, model_h=1080)
    bbox_pixels = [100, 100, 200, 200]

    # Point inside (150, 150)
    pt_inside = transformer.resolve_click_point([150, 150], bbox_pixels)
    assert pt_inside == (150, 150)

    # Point outside (300, 300) -> should safely fallback to center (150, 150)
    pt_outside = transformer.resolve_click_point([300, 300], bbox_pixels)
    assert pt_outside == (150, 150)
