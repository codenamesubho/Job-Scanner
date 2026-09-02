from scanner.ats_common import html_to_text


def test_html_to_text_strips_normal_markup():
    assert html_to_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_html_to_text_unescapes_double_encoded_content():
    """Greenhouse's job `content` field comes back HTML-escaped inside the
    JSON string itself (e.g. "&lt;div&gt;...") rather than as real markup —
    a single BeautifulSoup pass over that leaves literal "<div>"-looking
    text in the output instead of stripping it as a tag."""
    double_encoded = "&lt;div class=&quot;content-intro&quot;&gt;&lt;p&gt;Hello world&lt;/p&gt;&lt;/div&gt;"

    text = html_to_text(double_encoded)

    assert text == "Hello world"
    assert "<div" not in text
    assert "<p>" not in text


def test_html_to_text_empty_input():
    assert html_to_text("") == ""
