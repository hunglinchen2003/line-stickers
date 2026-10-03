"""Turn magenta-background sticker art into a LINE Creators upload pack."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ASSETS = Path(r"C:\Users\hung-lin\.cursor\projects\c-Users-hung-lin-Desktop-line\assets")
OUT = Path(r"C:\Users\hung-lin\Desktop\line貼圖\01-AI說的就對了")

SOURCES = [
    "sticker-01-ai-shuo-de.jpg",
    "sticker-02-ai-renzheng.jpg",
    "sticker-03-wo-wen-ai.jpg",
    "sticker-04-ai-yiding-dui.jpg",
    "sticker-05-rang-ai-xiang.jpg",
    "sticker-06-ai-shuo-keyi.jpg",
    "sticker-07-ai-shuo-buxing.jpg",
    "sticker-08-zai-wen-yici.jpg",
    "sticker-09-ai-bu-zhidao.jpg",
    "sticker-10-qu-wen-ai.jpg",
    "sticker-11-fuzhi-tieshang.jpg",
    "sticker-12-ai-dangji.jpg",
    "sticker-13-xiexie-ai.jpg",
    "sticker-14-ting-ai-de.jpg",
    "sticker-15-jiao-gei-ai.jpg",
    "sticker-16-wo-xin-ai.jpg",
]


def key_magenta(im: Image.Image) -> Image.Image:
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    corners = np.stack(
        [rgb[2, 2], rgb[2, w - 3], rgb[h - 3, 2], rgb[h - 3, w - 3]],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=2)

    # Flat magenta field is the background. Keep a soft edge so outlines stay clean.
    lo, hi = 36.0, 78.0
    alpha = np.clip((dist - lo) / (hi - lo), 0.0, 1.0)

    # Only treat near-magenta pixels as removable, so pink blush and hearts stay.
    magenta = (rgb[:, :, 0] > 160) & (rgb[:, :, 2] > 120) & (rgb[:, :, 1] + 35 < rgb[:, :, 0])
    alpha = np.where(magenta, alpha, 1.0)
    alpha = np.where(dist < lo, 0.0, alpha)

    a = alpha[..., None]
    # Unmix the magenta that JPEG left on the outline.
    straight = (rgb - bg * (1.0 - a)) / np.maximum(a, 1e-3)
    straight = np.clip(straight, 0, 255)

    out = np.dstack([straight, alpha * 255.0]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def content_bbox(im: Image.Image, thresh: int = 12):
    alpha = np.asarray(im)[:, :, 3]
    ys, xs = np.where(alpha > thresh)
    if len(xs) == 0:
        return (0, 0, im.width, im.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def fit(im: Image.Image, size, margin: int) -> Image.Image:
    w, h = size
    box = content_bbox(im)
    cropped = im.crop(box)
    max_w, max_h = w - margin * 2, h - margin * 2
    scale = min(max_w / cropped.width, max_h / cropped.height)
    nw = max(2, int(cropped.width * scale) // 2 * 2)
    nh = max(2, int(cropped.height * scale) // 2 * 2)
    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2), resized)
    return canvas


def save_png(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG", dpi=(72, 72), optimize=True)


def checker(im: Image.Image, cell: int = 16) -> Image.Image:
    bg = Image.new("RGB", im.size, (230, 230, 230))
    draw = ImageDraw.Draw(bg)
    for y in range(0, im.height, cell):
        for x in range(0, im.width, cell):
            if (x // cell + y // cell) % 2 == 0:
                draw.rectangle([x, y, x + cell, y + cell], fill=(200, 200, 200))
    bg.paste(im, mask=im.split()[-1])
    return bg


def main() -> None:
    sticker_dir = OUT / "貼圖"
    keyed = []
    for i, name in enumerate(SOURCES, start=1):
        src = key_magenta(Image.open(ASSETS / name))
        sticker = fit(src, (370, 320), margin=8)
        save_png(sticker, sticker_dir / f"{i:02d}.png")
        keyed.append(src)
        print(f"{i:02d}.png {sticker_dir.joinpath(f'{i:02d}.png').stat().st_size}")

    main_im = fit(keyed[0], (240, 240), margin=6)
    save_png(main_im, OUT / "main.png")

    # Tab image is the face, cropped from the upper part of the first sticker.
    box = content_bbox(keyed[0])
    x0, y0, x1, y1 = box
    face = keyed[0].crop((x0, y0, x1, y0 + int((y1 - y0) * 0.62)))
    tab = fit(face, (96, 74), margin=2)
    save_png(tab, OUT / "tab.png")

    # Contact sheet for visual check.
    sheet = Image.new("RGB", (370 * 4, 320 * 4), (255, 255, 255))
    for i in range(16):
        tile = checker(Image.open(sticker_dir / f"{i+1:02d}.png"), cell=12)
        sheet.paste(tile, ((i % 4) * 370, (i // 4) * 320))
    sheet.save(OUT / "_preview.jpg", quality=90)
    checker(main_im, 10).save(OUT / "_preview_main.jpg", quality=90)
    checker(tab.resize((96 * 4, 74 * 4), Image.Resampling.NEAREST), 8).save(
        OUT / "_preview_tab.jpg", quality=90
    )
    print("main", (OUT / "main.png").stat().st_size, "tab", (OUT / "tab.png").stat().st_size)


if __name__ == "__main__":
    main()
