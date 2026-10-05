from pathlib import Path

import pytest

from minutes.errors import PermanentStageError
from minutes.ingest.captions import Cue, Segment, parse_srt, parse_vtt, segment_cues


def test_parse_srt_reads_times_and_joins_lines() -> None:
    data = (
        b"1\n00:00:03,303 --> 00:00:13,513\n[MUSIC]\n[MUSIC]\n\n"
        b"2\n00:00:18,785 --> 00:00:19,185\n-- WHERE\n"
    )
    assert parse_srt(data) == [Cue(3303, 13513, "[MUSIC] [MUSIC]"), Cue(18785, 19185, "-- WHERE")]
    assert parse_srt(
        b"1\n1:02:03.004 --> 1:02:04.005\n123\n2\n1:02:04.005 --> 1:02:05.006\nNEXT\n"
    ) == [Cue(3723004, 3724005, "123"), Cue(3724005, 3725006, "NEXT")]


def test_parse_vtt_skips_header_and_unescapes_entities() -> None:
    data = (
        b"WEBVTT\r\n\r\nNOTE a comment\r\n00:00:00.000 --> 00:00:01.000\r\nignored\r\n\r\n"
        b"STYLE\r\n::cue { color: yellow; }\r\n\r\n"
        b"cue-name\r\n00:00:03.303 --> 00:00:13.513 align:start position:0%\r\n"
        b"O&#039;CLOCK <c.yellow>text</c>\r\n\r\n"
        b"00:00:14.000 --> 00:00:15.000\r\n<c.yellow></c>\r\n\r\n"
        b"00:00:16.000 --> 00:00:17.000\r\n\xff &nbsp; &#x201C;hi&#x201D;\r\n"
    )
    assert parse_vtt(data) == [
        Cue(3303, 13513, "O'CLOCK text"),
        Cue(16000, 17000, '\ufffd "hi"'),
    ]


def test_malformed_timing_line_is_skipped() -> None:
    data = (
        b"00:00:00,000 --> 00:00:01,000\nFIRST\n"
        b"00:00:bad --> 00:00:02,000\nMUST BE SKIPPED\n\n"
        b"00:00:03,000 --> 00:00:04,000\nLAST\n"
    )
    for parser in (parse_srt, parse_vtt):
        assert parser(data) == [Cue(0, 1000, "FIRST"), Cue(3000, 4000, "LAST")]


def test_no_cues_is_permanent_error() -> None:
    for parser in (parse_srt, parse_vtt):
        for data in (b"", b"WEBVTT\n\ninvalid\n", b"00:00:00.000 --> 00:00:01.000\n<c></c>\n"):
            with pytest.raises(PermanentStageError, match=r"^no cues$"):
                parser(data)


def test_segment_closes_after_60_seconds() -> None:
    cues = [Cue(0, 30000, "first"), Cue(30000, 59000, "second"), Cue(59000, 61000, "third")]
    assert segment_cues(cues) == [
        Segment(1, 0, 59000, "first second", None),
        Segment(2, 59000, 61000, "third", None),
    ]
    assert segment_cues([*cues[:2], Cue(59000, 60000, "third")]) == [
        Segment(1, 0, 60000, "first second third", None)
    ]


def test_segment_closes_at_1200_characters() -> None:
    cues = [Cue(0, 1000, "a" * 500), Cue(1000, 2000, "b" * 500), Cue(2000, 3000, "c" * 500)]
    assert segment_cues(cues) == [
        Segment(1, 0, 2000, "a" * 500 + " " + "b" * 500, None),
        Segment(2, 2000, 3000, "c" * 500, None),
    ]
    assert segment_cues([Cue(0, 1000, "a" * 699), cues[1]]) == [
        Segment(1, 0, 2000, "a" * 699 + " " + "b" * 500, None)
    ]


def test_segment_closes_on_speaker_marker_after_300_characters() -> None:
    for marker in ("SPEAKER 2 (COUNCIL): ROLL CALL", "-- NEXT", ">> NEXT"):
        cue = Cue(1000, 2000, marker)
        speaker = "SPEAKER 2 (COUNCIL)" if marker.startswith("SPEAKER") else None
        for length in (300, 310):
            assert segment_cues([Cue(0, 1000, "a" * length), cue]) == [
                Segment(1, 0, 1000, "a" * length, None),
                Segment(2, 1000, 2000, marker, speaker),
            ]
        assert segment_cues([Cue(0, 1000, "a" * 290), cue]) == [
            Segment(1, 0, 2000, "a" * 290 + " " + marker, None)
        ]


def test_segment_closes_on_gap_over_5_seconds() -> None:
    assert segment_cues([Cue(0, 1000, "first"), Cue(6001, 7000, "second")]) == [
        Segment(1, 0, 1000, "first", None),
        Segment(2, 6001, 7000, "second", None),
    ]
    assert segment_cues([Cue(0, 1000, "first"), Cue(6000, 7000, "second")]) == [
        Segment(1, 0, 7000, "first second", None)
    ]


def test_long_cue_is_split_every_1200_characters() -> None:
    text = "a" * 1200 + "b" * 1200 + "c" * 100
    assert segment_cues([Cue(1234, 5678, text)]) == [
        Segment(1, 1234, 5678, "a" * 1200, None),
        Segment(2, 1234, 5678, "b" * 1200, None),
        Segment(3, 1234, 5678, "c" * 100, None),
    ]


def test_speaker_label_from_first_cue() -> None:
    for text, speaker in (
        ("SPEAKER 2 (COUNCIL): ROLL CALL", "SPEAKER 2 (COUNCIL)"),
        ("-- GOOD AFTERNOON", None),
        ("-- COUNCILMEMBER FLORES: I WILL", "COUNCILMEMBER FLORES"),
        (">> COUNCILMEMBER FLORES: I WILL", "COUNCILMEMBER FLORES"),
    ):
        assert segment_cues([Cue(0, 1000, text), Cue(1000, 2000, "CHAIR: NEXT")]) == [
            Segment(1, 0, 2000, text + " CHAIR: NEXT", speaker)
        ]


def test_segmentation_is_deterministic() -> None:
    corpus = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"
    for name, parser in (
        ("alder-transcript-101.srt", parse_srt),
        ("alder-transcript-102.srt", parse_srt),
        ("birch-transcript-201.vtt", parse_vtt),
        ("birch-transcript-202.vtt", parse_vtt),
    ):
        data = (corpus / name).read_bytes()
        first = segment_cues(parser(data))
        second = segment_cues(parser(data))
        assert first == second
        assert first
        assert [segment.index for segment in first] == list(range(1, len(first) + 1))
        assert all(len(segment.text) <= 1200 for segment in first)
        assert all(segment.end_ms - segment.start_ms <= 60000 for segment in first)
