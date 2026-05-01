"""Gen Image Plugin for AstrBot.

Supports:
- :draw <prompt> [-b <negative>] [-s <style>] [-ar <aspect>] [-m <model>] [-n <count>]
- :pdraw <prompt> [-b <negative>] [-s <style>] [-m <model>] (with an attached image)
"""

import argparse
import re
from typing import Any

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


@register(
    "astrbot_plugin_gen_image",
    "AstrBot",
    "AI Image Generation Plugin. Use :draw for text-to-image, :pdraw for image-to-image.",
    "1.0.0",
)
class GenImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        openai_cfg = config.get("openai", {})
        self.api_key = str(openai_cfg.get("api_key", "")).strip()
        self.api_base = str(
            openai_cfg.get("api_base", "https://api.openai.com/v1")
        ).strip()
        self.model = str(openai_cfg.get("model", "dall-e-3")).strip()
        self.default_size = str(openai_cfg.get("default_size", "1024x1024")).strip()
        self.default_quality = str(
            openai_cfg.get("default_quality", "standard")
        ).strip()
        self.default_num = int(openai_cfg.get("default_num", 1))
        self.timeout = int(openai_cfg.get("timeout", 120))

    # ------------------------------------------------------------
    # Argument parser helpers
    # ------------------------------------------------------------

    @staticmethod
    def _build_argparser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("prompt", nargs="?", default="")
        parser.add_argument("-b", "--negative", default="", help="Negative prompt")
        parser.add_argument("-s", "--style", default="", help="Style preset")
        parser.add_argument(
            "-ar", "--aspect", default="", help="Aspect ratio, e.g. 16:9"
        )
        parser.add_argument("-m", "--model", default="", help="Model override")
        parser.add_argument("-n", "--num", type=int, default=0, help="Number of images")
        return parser

    @staticmethod
    def _parse_draw_args(
        message_str: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Parse :draw/:pdraw arguments from the message string.

        Returns (prompt, kwargs) on success, or None if the command doesn't match.
        """
        text = message_str.strip()
        matched = re.match(r"^:(draw|pdraw)\b\s*", text)
        if not matched:
            return None
        remainder = text[matched.end() :].strip()

        parser = GenImagePlugin._build_argparser()
        try:
            args = parser.parse_args(remainder.split())
        except (SystemExit, ValueError):
            return None

        kwargs: dict[str, Any] = {}
        prompt = args.prompt
        if args.negative:
            kwargs["negative_prompt"] = args.negative
        if args.style:
            kwargs["style"] = args.style
        if args.aspect:
            kwargs["aspect_ratio"] = args.aspect
        if args.model:
            kwargs["model"] = args.model
        if args.num and args.num > 0:
            kwargs["num_images"] = args.num

        return prompt, kwargs

    @staticmethod
    def _aspect_to_size(aspect: str) -> str:
        """Convert aspect ratio string to a standard size supported by DALL-E."""
        mapping = {
            "1:1": "1024x1024",
            "16:9": "1792x1024",
            "9:16": "1024x1792",
            "4:3": "1024x768",
            "3:4": "768x1024",
            "3:2": "1216x832",
            "2:3": "832x1216",
        }
        normalized = aspect.strip().replace(" ", "")
        return mapping.get(normalized, "1024x1024")

    # ------------------------------------------------------------
    # Image generation backend
    # ------------------------------------------------------------

    async def _generate_openai(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        model: str = "",
        size: str = "",
        quality: str = "",
        num_images: int = 1,
    ) -> list[str]:
        """Generate images via OpenAI / DALL-E compatible API."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not configured. Set api_key in plugin config."
            )

        url = f"{self.api_base.rstrip('/')}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "n": num_images,
            "size": size or self.default_size,
            "quality": quality or self.default_quality,
        }
        if negative_prompt:
            # Some compatible providers support negative_prompt
            payload["negative_prompt"] = negative_prompt

        logger.debug(
            "OpenAI image request: url=%s model=%s size=%s",
            url,
            payload["model"],
            payload["size"],
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

    @filter.regex(r"^:draw\b")
    async def draw_command(self, event: AstrMessageEvent):
        """Text-to-image generation."""
        result = self._parse_draw_args(event.get_message_str())
        if result is None:
            return
        prompt, kwargs = result

        if not prompt:
            yield event.plain_result(
                "Usage: :draw <prompt> [-b <negative>] [-s <style>] [-ar <aspect>] [-m <model>] [-n <count>]"
            )
            return

        yield event.plain_result("🎨 Generating image, please wait...")

        num = kwargs.pop("num_images", 0)
        if num <= 0:
            num = self.default_num
        style = kwargs.pop("style", "")
        aspect_ratio = kwargs.pop("aspect_ratio", "")
        model = kwargs.get("model", "")
        negative = kwargs.get("negative_prompt", "")

        if style:
            prompt = f"{prompt}, style: {style}"

        size = self._aspect_to_size(aspect_ratio) if aspect_ratio else self.default_size

        try:
            urls = await self._generate_openai(
                prompt,
                negative_prompt=negative,
                model=model,
                size=size,
                num_images=num,
            )
        except Exception as e:
            logger.exception(f"Image generation failed: {e}")
            yield event.plain_result(f"❌ Image generation failed: {e}")
            return

        for url in urls:
            yield event.image_result(url)

    @filter.regex(r"^:pdraw\b")
    async def pdraw_command(self, event: AstrMessageEvent):
        """Image-to-image generation."""
        result = self._parse_draw_args(event.get_message_str())
        if result is None:
            return
        prompt, kwargs = result

        # Extract attached images
        images = [c for c in event.get_messages() if isinstance(c, Comp.Image)]
        if not images:
            yield event.plain_result(
                "Usage: :pdraw <prompt> [-b <negative>] [-s <style>] [-m <model>] (attach an image)"
            )
            return

        if not prompt:
            yield event.plain_result(
                "Please provide a prompt for image-to-image generation."
            )
            return

        yield event.plain_result("🎨 Processing image-to-image, please wait...")

        num = kwargs.get("num_images", 0)
        if num <= 0:
            num = self.default_num

        negative = kwargs.get("negative_prompt", "")
        style = kwargs.get("style", "")

        if style:
            prompt = f"{prompt}, style: {style}"

        try:
            urls = await self._generate_openai(
                prompt,
                negative_prompt=negative,
                num_images=num,
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
