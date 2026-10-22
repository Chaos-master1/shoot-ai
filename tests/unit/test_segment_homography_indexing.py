"""Regression tests for per-segment homography indexing (2026-09-02 audit).

The producer (CVService.process_video) numbers camera segments by
incrementing current_segment when it reaches the first detection frame at
or after a cut. The consumer (the physical-profiles block) computes
seg = number of cuts at-or-before each frame. An old special case
(or frame_number == 0 and cuts exist) pre-incremented the producer to
segment 1 at frame 0, shifting the entire per-segment homography map by
one -- every frame after cut N was measured with the PREVIOUS angle's
homography.

These tests pin the shared indexing contract both sides must follow.
"""

from __future__ import annotations


def segment_for_frame(frame_number: int, sorted_cuts: list[int]) -> int:
    """Consumer-side segment index: number of cuts at-or-before the frame."""
    return sum(1 for c in sorted_cuts if c <= frame_number)


class TestSegmentIndexingContract:
    def test_no_cuts_single_segment(self):
        assert segment_for_frame(0, []) == 0
        assert segment_for_frame(999, []) == 0

    def test_frames_before_first_cut_are_segment_0(self):
        cuts = [100, 400]
        # Frame 0 must remain segment 0 -- the old special case bumped it
        # to 1 whenever ANY cut existed anywhere in the video.
        assert segment_for_frame(0, cuts) == 0
        assert segment_for_frame(99, cuts) == 0

    def test_frames_between_cuts_track_their_segment(self):
        cuts = [100, 400]
        assert segment_for_frame(100, cuts) == 1
        assert segment_for_frame(399, cuts) == 1
        assert segment_for_frame(400, cuts) == 2
        assert segment_for_frame(1000, cuts) == 2

    def test_producer_and_consumer_agree_on_boundary_frames(self):
        """Simulate the producer's increment logic against the consumer's
        count over a frame stream with detection frames every frame_skip
        frames, and assert both agree on every frame."""
        frame_skip = 3
        cuts = [10, 22, 61]
        sorted_cuts = sorted(cuts)
        current_segment = 0
        prev_det_frame = -frame_skip
        producer_segments: dict[int, int] = {}
        for fn in range(0, 70):
            if fn % frame_skip == 0:
                cut_hit = any(c > prev_det_frame and c <= fn for c in cuts)
                if cut_hit:
                    current_segment += 1
                producer_segments[fn] = current_segment
                prev_det_frame = fn
            # Consumer-side evaluation for every frame (lookup happens
            # post-processing over all recorded frames):
            consumer_seg = segment_for_frame(fn, sorted_cuts)
            if fn in producer_segments:
                # On detection frames, producer's own segment must equal
                # the consumer count for that frame.
                assert producer_segments[fn] == consumer_seg, (
                    f"frame {fn}: producer segment {producer_segments[fn]} "
                    f"!= consumer segment {consumer_seg}"
                )
