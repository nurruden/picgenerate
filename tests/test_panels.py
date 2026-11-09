from PIL import Image

from generatepic.panels import split_panels


def test_collage_splits_into_each_panel(tmp_path):
    image = Image.new("RGB", (220, 180), "white")
    image.paste(Image.new("RGB", (90, 70), (20, 40, 80)), (8, 8))
    image.paste(Image.new("RGB", (90, 70), (80, 30, 20)), (122, 8))
    image.paste(Image.new("RGB", (204, 70), (20, 80, 40)), (8, 100))
    source = tmp_path / "sheet.png"
    image.save(source)

    panels = split_panels(source)

    assert len(panels) == 3
    assert [panel.name for panel in panels] == ["sheet-1.png", "sheet-2.png", "sheet-3.png"]
    assert Image.open(panels[0]).size == (90, 70)


def test_two_pixel_gaps_still_split(tmp_path):
    image = Image.new("RGB", (120, 104), "white")
    image.paste(Image.new("RGB", (120, 50), (10, 20, 30)), (0, 0))
    image.paste(Image.new("RGB", (59, 52), (40, 10, 10)), (0, 52))
    image.paste(Image.new("RGB", (59, 52), (10, 40, 10)), (61, 52))
    source = tmp_path / "thin.png"
    image.save(source)

    panels = split_panels(source)

    assert len(panels) == 3
    assert [Image.open(panel).size for panel in panels] == [(120, 50), (59, 52), (59, 52)]


def test_light_gray_gap_splits(tmp_path):
    image = Image.new("RGB", (160, 120), (242, 242, 242))
    image.paste(Image.new("RGB", (160, 50), (20, 30, 40)), (0, 0))
    image.paste(Image.new("RGB", (70, 67), (80, 20, 20)), (0, 53))
    image.paste(Image.new("RGB", (86, 67), (20, 80, 30)), (74, 53))
    source = tmp_path / "gray.png"
    image.save(source)

    assert len(split_panels(source)) == 3


def test_single_photo_stays_whole(tmp_path):
    image = Image.new("RGB", (80, 60), (30, 90, 140))
    source = tmp_path / "photo.png"
    image.save(source)

    assert split_panels(source) == [source]
