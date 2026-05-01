"""Gen Image Plugin for AstrBot.

Supports:
- :draw -p <prompt> [-pre <preset>] [-s <size>] [-q <quality>] [-b <background>]
- :pdraw -p <prompt> [-pre <preset>] [-s <size>] [-q <quality>] [-b <background>] (with an attached image)

At least one of `-p` or `-pre` is required.
"""

import argparse
from typing import Any

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star


class GenImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # OpenAI compatible config
        openai_cfg = config.get("openai", {})
        self.api_key = str(openai_cfg.get("api_key", "")).strip()
        self.txt2img_url = str(
            openai_cfg.get(
                "txt2img_url", "https://api.openai.com/v1/images/generations"
            )
        ).strip()
        self.img2img_url = str(
            openai_cfg.get("img2img_url", "https://api.openai.com/v1/images/edits")
        ).strip()
        self.model = str(openai_cfg.get("model", "dall-e-3")).strip()
        self.default_size = str(openai_cfg.get("default_size", "2048x2048")).strip()
        self.default_quality = str(openai_cfg.get("default_quality", "medium")).strip()
        self.default_background = str(
            openai_cfg.get("default_background", "auto")
        ).strip()
        self.default_num = int(openai_cfg.get("default_num", 1))
        self.timeout = int(openai_cfg.get("timeout", 120))

        # Preset prompts: key=value per line
        presets_cfg = config.get("presets", {})
        preset_text = str(presets_cfg.get("preset_list", "")).strip()
        self.presets: dict[str, str] = {}
        if preset_text:
            for line in preset_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    self.presets[key.strip()] = value.strip()

    # ------------------------------------------------------------
    # Argument parser helpers
    # ------------------------------------------------------------

    SIZES = [
        "1024x1024",
        "1024x1536",
        "1536x1024",
        "2048x2048",
        "2048x1152",
        "3840x2160",
        "2160x3840",
    ]
    QUALITIES = ["low", "medium", "high"]
    BACKGROUNDS = ["transparent", "opaque", "auto"]

    @staticmethod
    def _build_argparser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-p", "--prompt", default="", help="Image prompt")
        parser.add_argument("-pre", "--preset", default="", help="Preset prompt key")
        parser.add_argument(
            "-s",
            "--size",
            default="2048x2048",
            choices=GenImagePlugin.SIZES,
            help="Image size",
        )
        parser.add_argument(
            "-q",
            "--quality",
            default="medium",
            choices=GenImagePlugin.QUALITIES,
            help="Image quality",
        )
        parser.add_argument(
            "-b",
            "--background",
            default="auto",
            choices=GenImagePlugin.BACKGROUNDS,
            help="Background mode",
        )
        return parser

    def _resolve_prompt(self, preset_key: str, user_prompt: str) -> str | None:
        """Resolve the final prompt from preset key and/or user prompt.

        Returns the resolved prompt string, or None if neither is provided.
        """
        result_parts: list[str] = []
        if preset_key:
            preset_text = self.presets.get(preset_key, "")
            if not preset_text:
                return None  # invalid preset
            result_parts.append(preset_text)
        if user_prompt:
            result_parts.append(user_prompt)
        return ", ".join(result_parts) if result_parts else None

    def _parse_named_args(self, raw_args: str) -> argparse.Namespace | None:
        """Parse named arguments from the raw command args string."""
        parser = self._build_argparser()
        try:
            return parser.parse_args(raw_args.split())
        except (SystemExit, ValueError):
            return None

    # ------------------------------------------------------------
    # Image generation backend
    # ------------------------------------------------------------

    async def _generate_openai(
        self,
        url: str,
        prompt: str,
        *,
        size: str = "",
        quality: str = "",
        background: str = "",
        num_images: int = 1,
    ) -> list[str]:
        """Generate images via OpenAI / DALL-E compatible API."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not configured. Set api_key in plugin config."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": num_images,
            "size": size or self.default_size,
            "quality": quality or self.default_quality,
            "background": background or self.default_background,
            "output_format": "png",
        }

        logger.debug(
            "OpenAI image request: url=%s model=%s size=%s quality=%s",
            url,
            payload["model"],
            payload["size"],
            payload["quality"],
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"OpenAI API error ({resp.status}): {error_text}"
                    )
                data = await resp.json()

        urls: list[str] = []
        for item in data.get("data", []):
            url_or_b64 = item.get("url") or item.get("b64_json", "")
            if url_or_b64:
                urls.append(url_or_b64)
        return urls

    # ------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------

    def _strip_command_prefix(self, message_str: str, command: str) -> str:
        """Strip the command prefix and return remaining args."""
        text = (message_str or "").strip()
        if not text:
            return ""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        if cmd in {command, f"/{command}", f":{command}"}:
            return parts[1].strip() if len(parts) > 1 else ""
        return text

    @filter.command("draw")
    async def draw_command(self, event: AstrMessageEvent):
        """Text-to-image generation."""
        raw_args = self._strip_command_prefix(event.message_str, "draw")
        args = self._parse_named_args(raw_args)
        if args is None:
            yield event.plain_result(
                "Usage: :draw -p <prompt> [-pre <preset_key>] [-s <size>] [-q <quality>] [-b <background>]\n"
                "At least one of -p or -pre is required."
            )
            return

        prompt = self._resolve_prompt(args.preset, args.prompt)
        if not prompt:
            if args.preset:
                yield event.plain_result(f"❌ Unknown preset key: {args.preset}")
            else:
                yield event.plain_result(
                    "Usage: :draw -p <prompt> [-pre <preset_key>] [-s <size>] [-q <quality>] [-b <background>]\n"
                    "At least one of -p or -pre is required."
                )
            return

        yield event.plain_result("🎨 Generating image, please wait...")

        try:
            urls = await self._generate_openai(
                self.txt2img_url,
                prompt,
                size=args.size,
                quality=args.quality,
                background=args.background,
                num_images=self.default_num,
            )
        except Exception as e:
            logger.exception(f"Image generation failed: {e}")
            yield event.plain_result(f"❌ Image generation failed: {e}")
            return

        for url in urls:
            yield event.image_result(url)

    @filter.command("pdraw")
    async def pdraw_command(self, event: AstrMessageEvent):
        """Image-to-image generation."""
        # Require an attached image
        images = [c for c in event.get_messages() if isinstance(c, Comp.Image)]
        if not images:
            yield event.plain_result(
                "Usage: :pdraw -p <prompt> [-pre <preset_key>] [-s <size>] [-q <quality>] [-b <background>] (attach an image)\n"
                "At least one of -p or -pre is required."
            )
            return

        raw_args = self._strip_command_prefix(event.message_str, "pdraw")
        args = self._parse_named_args(raw_args)
        if args is None:
            yield event.plain_result(
                "Usage: :pdraw -p <prompt> [-pre <preset_key>] [-s <size>] [-q <quality>] [-b <background>] (attach an image)\n"
                "At least one of -p or -pre is required."
            )
            return

        prompt = self._resolve_prompt(args.preset, args.prompt)
        if not prompt:
            if args.preset:
                yield event.plain_result(f"❌ Unknown preset key: {args.preset}")
            else:
                yield event.plain_result("At least one of -p or -pre is required.")
            return

        yield event.plain_result("🎨 Processing image-to-image, please wait...")

        try:
            urls = await self._generate_openai(
                self.img2img_url,
                prompt,
                size=args.size,
                quality=args.quality,
                background=args.background,
                num_images=self.default_num,
            )
        except Exception as e:
            logger.exception(f"Image generation failed: {e}")
            yield event.plain_result(f"❌ Image generation failed: {e}")
            return

        for url in urls:
            yield event.image_result(url)

    async def terminate(self):
        """Cleanup on plugin unload."""
        pass
