from pathlib import Path


def main(root: str | Path | None = None) -> None:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    splits_root = repo_root / "splits"
    splits_root.mkdir(exist_ok=True)

    for split in ("train", "val", "test"):
        image_dir = repo_root / split / "images"
        mask_dir = repo_root / split / "masks"

        if not image_dir.exists() or not mask_dir.exists():
            print(f"{split}: skipped (missing dataset folders)")
            continue

        image_stems = sorted(path.stem for path in image_dir.glob("*.jpg"))
        mask_stems = {path.stem for path in mask_dir.glob("*.png")}

        valid_stems = [stem for stem in image_stems if stem in mask_stems]
        missing_masks = [stem for stem in image_stems if stem not in mask_stems]
        extra_masks = sorted(mask_stems - set(image_stems))

        split_file = splits_root / f"{split}.txt"
        contents = "\n".join(valid_stems)
        split_file.write_text(f"{contents}\n" if contents else "", encoding="utf-8")

        print(
            f"{split}: {len(valid_stems)} valid pairs, "
            f"{len(missing_masks)} images without masks, "
            f"{len(extra_masks)} extra masks"
        )

        if missing_masks:
            print(f"  missing mask sample: {missing_masks[:5]}")
        if extra_masks:
            print(f"  extra mask sample: {extra_masks[:5]}")


if __name__ == "__main__":
    main()
