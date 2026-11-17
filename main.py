import argparse
from pathlib import Path

import cv2

from tracker import ObjectTracker, draw_box

VIDEO_DIR = Path(__file__).resolve().parent.parent / "test-videos"
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def videos_in_dir(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"нет папки {directory}")
    found = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not found:
        raise FileNotFoundError(f"в {directory} нет видео")
    return found


def resolve_video(name):
    path = Path(name)
    if path.is_file():
        return path
    candidate = VIDEO_DIR / name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"нет видео {name}. Искал этот путь и {candidate}")


def open_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"не открывается видео {path}")
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"в видео нет кадров: {path}")
    return capture, frame


def track(input_path, output_path, label):
    capture, template = open_video(input_path)
    tracker = ObjectTracker(template)
    height, width = template.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 15
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"не удалось записать {output_path}")

    seen = 0
    missed = 0
    frame = template
    while frame is not None:
        corners = tracker.locate(frame)
        if corners is None:
            writer.write(frame)
            missed += 1
        else:
            writer.write(draw_box(frame, corners, label))
            seen += 1
        ok, frame = capture.read()
        if not ok:
            frame = None
    capture.release()
    writer.release()
    print(f"{input_path.name}: кадров с рамкой {seen}, без рамки {missed}")
    return seen, missed


def save_previews(video_path, directory):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"не открывается результат {video_path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 1:
        capture.release()
        raise RuntimeError(f"в результате нет кадров: {video_path}")
    indexes = sorted({0, count // 2, count - 1})
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for index in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        if index not in indexes:
            continue
        path = directory / f"{video_path.stem}_{index:03d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"не удалось сохранить {path}")
        saved.append(path)
    capture.release()
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Трекинг объекта по ключевым точкам SIFT")
    parser.add_argument(
        "--input",
        default=None,
        help="файл или имя ролика из test-videos; без аргумента читается вся папка",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="object")
    parser.add_argument("--previews", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input:
        inputs = [resolve_video(args.input)]
    else:
        inputs = videos_in_dir(VIDEO_DIR)
    previews = Path(args.previews)
    for input_path in inputs:
        if args.output and len(inputs) == 1:
            output_path = Path(args.output)
        else:
            output_path = previews / f"{input_path.stem}.mp4"
        track(input_path, output_path, args.label)
        for path in save_previews(output_path, previews):
            print(path)


if __name__ == "__main__":
    main()
