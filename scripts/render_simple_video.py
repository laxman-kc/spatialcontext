"""Render a short, one-question edit from the verified video-test evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import matplotlib
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/video-demo/input/source.mp4')
    parser.add_argument('--frame', type=Path, default=ROOT / 'artifacts/video-demo/run/full/frame_0003.png')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/video-demo/simple/before-after.mp4')
    args = parser.parse_args()
    evidence = ROOT / 'results/video-demo/data.json'
    data = json.loads(evidence.read_text())
    question = data['questions'][0]
    if question['id'] != 'original-8918' or question['gold'] != 'B':
        raise ValueError('This edit is specific to the recorded first question.')
    if sha256(args.source) != data['source']['sha256']:
        raise ValueError('Source video does not match the saved test.')
    if sha256(args.frame) != data['source']['prepared_png_sha256'][3]:
        raise ValueError('Remembered frame does not match the saved test.')
    for mode, expected, correct in [('full', 'A', False), ('memory_base', 'B', True)]:
        result = question['results'][mode]
        if result['prediction'] != expected or result['correct'] != correct:
            raise ValueError('Recorded answers differ from this presentation.')
    fonts = {
        size: ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'), size)
        for size in (23, 26, 30, 44, 46)
    }
    with av.open(str(args.source)) as source:
        frames = [frame.to_image() for frame in source.decode(video=0)]
    remembered = Image.open(args.frame).convert('RGB')
    width, height = remembered.size
    crop = (int(width * .39), int(height * .32), int(width * .75), int(height * .80))
    zoom = remembered.crop(crop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previews = []
    output = av.open(str(args.output), 'w', options={'movflags': '+faststart'})
    stream = output.add_stream('libx264', rate=25)
    stream.width, stream.height, stream.pix_fmt = 1600, 900, 'yuv420p'
    stream.options = {'crf': '20', 'preset': 'medium'}
    for index in range(16 * 25):
        t = index / 25
        canvas = Image.new('RGB', (1600, 900), '#ffffff')
        draw = ImageDraw.Draw(canvas)

        def text(x, y, value, size=26, color='#172b3a'):
            draw.text((x, y), value, font=fonts[size], fill=color)

        text(44, 34, 'In clip 2, what was to the right of the truck?', 46)
        if t < 4:
            photo = frames[int((5.375 + t) * 24)]
            phase = 'Earlier scene · Clip 2'
        elif t < 6:
            photo = frames[int((12.5 + t - 4) * 24)]
            phase = 'Later footage'
        else:
            photo = zoom
            phase = 'Return to the earlier scene · Zoomed for visibility'
        text(44, 106, phase, 26)
        scale = min(1024 / photo.width, 576 / photo.height)
        photo = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.Resampling.LANCZOS)
        px, py = 44 + (1024 - photo.width) // 2, 160 + (576 - photo.height) // 2
        canvas.paste(photo, (px, py))
        if t >= 6:
            # Boxes annotate the presentation crop; they are never model inputs.
            def box(bounds, label, color):
                x1, y1, x2, y2 = bounds
                coords = (
                    px + (width * x1 - crop[0]) * photo.width / zoom.width,
                    py + (height * y1 - crop[1]) * photo.height / zoom.height,
                    px + (width * x2 - crop[0]) * photo.width / zoom.width,
                    py + (height * y2 - crop[1]) * photo.height / zoom.height,
                )
                draw.rectangle(coords, outline=color, width=3)
                tx, ty = coords[0], coords[1] - 32
                label_bounds = draw.textbbox((tx + 6, ty), label, font=fonts[23])
                draw.rectangle((tx, ty - 2, label_bounds[2] + 6, ty + 30), fill='#172b3a')
                text(tx + 6, ty, label, 23, color)
            box((.530, .403, .576, .619), 'Truck', '#ffffff')
            if t >= 10:
                box((.616, .58, .676, .69), 'Lights', '#92f3b8')
        for label, answer, verdict, y, visible, color in [
            ('Before memory', 'White car', 'Incorrect', 238, t >= 6, '#ad3836'),
            ('After memory', 'Spherical lights', 'Correct', 466, t >= 10, '#087844'),
        ]:
            text(1110, y, label, 30)
            text(1110, y + 55, answer if visible else '—', 44, color if visible else '#9ba7af')
            if visible:
                text(1110, y + 115, verdict, 26, color)
        draw.line((44, 789, 1556, 789), fill='#dbe1e5', width=2)
        text(44, 816, 'Same base model · Recorded answers · One example', 26, '#52616d')
        video_frame = av.VideoFrame.from_image(canvas)
        for packet in stream.encode(video_frame):
            output.mux(packet)
        if index in (25, 125, 200, 350):
            preview = args.output.parent / f'review-{index:03d}.png'
            canvas.save(preview)
            previews.append(str(preview.relative_to(ROOT)))
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    receipt = {
        'schema': 'video-demo-simple-edit-v1',
        'source_results': 'results/video-demo/data.json',
        'source_results_sha256': sha256(evidence),
        'source_video_sha256': sha256(args.source),
        'remembered_frame_sha256': sha256(args.frame),
        'script_sha256': sha256(Path(__file__)),
        'question_id': question['id'],
        'display_question_shortened': True,
        'modes': ['full', 'memory_base'],
        'duration_seconds': 16,
        'fps': 25,
        'dimensions': [1600, 900],
        'edit': [
            {'output_seconds': [0, 4], 'source_seconds': [5.375, 9.375], 'speed': 1},
            {'output_seconds': [4, 6], 'source_seconds': [12.5, 14.5], 'speed': 1},
            {'output_seconds': [6, 16], 'source_frame_index': 215,
             'source_seconds': data['source']['sample_pts'][3], 'display_crop_xyxy': list(crop)},
        ],
        'scope': 'Edited replay of saved predictions, not a new inference run. Crop, boxes and shortened labels are presentation only. The model saw the original verified sampled frames.',
        'video': {'file': args.output.name, 'bytes': args.output.stat().st_size, 'sha256': sha256(args.output)},
        'local_review_frames': previews,
    }
    (args.output.parent / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
