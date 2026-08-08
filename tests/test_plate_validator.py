from pelak_khan.postprocessing.plate_validator import display_plate, validate_standard_plate


def test_valid_standard_plate():
    valid, reasons = validate_standard_plate("18ق26744", {"ق", "س"})
    assert valid is True
    assert reasons == []


def test_short_plate_rejected():
    valid, reasons = validate_standard_plate("152", {"ق", "س"})
    assert valid is False
    assert reasons == ["invalid_length_3"]


def test_display_plate():
    assert display_plate("18ق26744") == "18 ق 267 | 44"
