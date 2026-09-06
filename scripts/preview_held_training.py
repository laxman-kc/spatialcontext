"""Render exact bank target frames for boundary triage; creates no approvals."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ecqa.artifacts import atomic_json
from ecqa.data import prepared_clip_ranges, sample_frame_indices, sha256_file, write_jsonl


def main():
    root = Path('data')
    output = root / 'fullstudy/held-training'
    output.mkdir(exist_ok=True)
    queue = []
    for path in sorted((root / 'fullstudy/preparation/review').glob('*/review.json')):
        meta = json.loads(path.read_text())
        if not meta['id'].startswith('train:') or meta.get('automatic_exclusion_or_hold') != 'boundaries_require_review':
            continue
        scan_path = path.parent / 'source_scan.json'
        scan = json.loads(scan_path.read_text())
        ranges = scan['boundary_verification']['ranges']
        target = meta['target_clip']
        record = {'id': meta['id'], 'queue_index': meta['queue_index'], 'candidate_folder': str(path.parent),
                  'source_scan_sha256': sha256_file(scan_path), 'question': meta['question'],
                  'options': meta['options'], 'answer': meta['answer'], 'target_clip': target,
                  'clip_count': len(ranges), 'boundary_verification': scan['boundary_verification'],
                  'status': 'not_reviewed; preview is not an approval'}
        if target is None or target >= len(ranges):
            record['triage'] = 'unresolved_target' if target is None else 'decoded_final_target'
        else:
            indices = sample_frame_indices(ranges)
            start, end = prepared_clip_ranges(len(ranges))[target - 1]
            indices = indices[start:end]
            bank = {f['frame']: f for clip in scan['donor_bank'] for f in clip}
            width, height = scan['canvas_wh']
            panel = Image.new('RGB', (width * 2, (height + 22) * ((len(indices) + 1) // 2)), 'white')
            draw = ImageDraw.Draw(panel)
            for n, index in enumerate(indices):
                frame = bank[index]
                source = root / frame['path']
                if sha256_file(source) != frame['sha256']:
                    raise ValueError('Bank pixel mismatch')
                x, y = n % 2 * width, n // 2 * (height + 22)
                panel.paste(Image.open(source), (x, y))
                draw.text((x, y + height + 2), f'tentative target source frame {index}; boundaries unapproved', fill='black')
            target_path = output / f"{meta['queue_index']:04d}_{meta['id'].split('_')[-1]}.png"
            panel.save(target_path)
            record.update(triage='potential_earlier_requires_visual_boundary_and_label_review',
                          target_preview=str(target_path), source_frame_indices=indices)
        queue.append(record)
    write_jsonl(output / 'queue.jsonl', queue)
    atomic_json(output / 'summary.json', {'held_records': len(queue), 'potential_earlier': sum('target_preview' in r for r in queue),
                                         'status': 'actual reviews required; no records approved'})
    print(json.dumps({'held_records': len(queue), 'previews': sum('target_preview' in r for r in queue)}))


if __name__ == '__main__':
    main()
