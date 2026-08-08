from pelak_khan.postprocessing.plate_regions import lookup_plate_region


def test_iran_99_is_tehran():
    info = lookup_plate_region("56ی16199")
    assert info["known"] is True
    assert info["iran_code"] == "99"
    assert info["city"] == "تهران"


def test_iran_15_is_tabriz():
    info = lookup_plate_region("18ق26715")
    assert info["known"] is True
    assert info["city"] == "تبریز"
    assert info["province"] == "آذربایجان شرقی"


def test_short_text_is_unknown():
    info = lookup_plate_region("152")
    assert info["known"] is False
