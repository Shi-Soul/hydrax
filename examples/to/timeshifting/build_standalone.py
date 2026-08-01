"""Embed every local media file into one portable HTML file."""

import base64
import json
import re
from pathlib import Path

import hydra
from omegaconf import DictConfig

MIME_TYPES = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@hydra.main(version_base=None, config_path=".", config_name="standalone")
def main(cfg: DictConfig) -> None:
    """Inline all local src and poster references into the dashboard."""
    output_dir = Path(str(cfg.output_dir)).resolve()
    source_path = output_dir / "index.html"
    target_path = output_dir / str(cfg.output_file)
    html = source_path.read_text(encoding="utf-8")

    def inline_value(value: str) -> str:
        if value.startswith("data:"):
            return value
        media_path = output_dir / value
        mime = MIME_TYPES.get(media_path.suffix.lower())
        if not media_path.is_file() or mime is None:
            return value
        encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def inline_attribute(match: re.Match) -> str:
        return (
            f'{match.group(1)}="{inline_value(match.group(2))}"'
        )

    def inline_data(match: re.Match) -> str:
        payload = json.loads(match.group(2))
        for row in payload.get("aggregates", []):
            for field in ("video", "poster"):
                if row.get(field):
                    row[field] = inline_value(str(row[field]))
        encoded = json.dumps(
            payload, separators=(",", ":"), allow_nan=False
        ).replace("</", "<\\/")
        return match.group(1) + encoded + match.group(3)

    standalone = re.sub(
        r'(src|poster)="((?!data:)[^"]+)"',
        inline_attribute,
        html,
    )
    standalone = re.sub(
        r'(<script id="data" type="application/json">)(.*?)(</script>)',
        inline_data,
        standalone,
        flags=re.DOTALL,
    )
    target_path.write_text(standalone, encoding="utf-8")
    print(
        f"wrote {target_path} "
        f"({target_path.stat().st_size / (1024 * 1024):.2f} MiB)"
    )


if __name__ == "__main__":
    main()
