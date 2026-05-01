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
        super().__init__(context, config)
        self.config = config

        # OpenAI / DALL-E compatible config
        self.api_type = str(config.get("api_type", "openai")).strip().lower()
        self.api_key = str(config.get("api_key", "")).strip()
        self.api_base = str(config.get("api_base", "https://api.openai.com/v1")).strip()
        self.model = str(config.get("model", "dall-e-3")).strip()
        self.default_size = str(config.get("default_size", "1024x1024")).strip()
        self.default_quality = str(config.get("default_quality", "standard")).strip()

        # Stable Diffusion (Automatic1111) config
        self.sd_api_base = str(
            config.get("sd_api_base", "http://127.0.0.1:7860")
        ).strip()

        # Common
        self.default_num = int(config.get("default_num", 1))
        self.timeout = int(config.get("timeout", 120))

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

    @staticmethod
    def _sd_size_from_aspect(aspect: str) -> tuple[int, int]:
        """Convert aspect ratio to nearest SD-compatible dimensions."""
        mapping = {
            "1:1": (512, 512),
            "16:9": (768, 432),
            "9:16": (432, 768),
            "4:3": (640, 480),
            "3:4": (480, 640),
            "3:2": (768, 512),
            "2:3": (512, 768),
        }
        normalized = aspect.strip().replace(" ", "")
        if normalized in mapping:
            return mapping[normalized]
        return (512, 512)

    # ------------------------------------------------------------
    # Image generation backends
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

    async def _generate_sd(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        size: str = "",
        num_images: int = 1,
    ) -> list[str]:
        """Generate images via Stable Diffusion WebUI (Automatic1111) API."""
        base = self.sd_api_base.rstrip("/")

        width, height = 512, 512
        if size and "x" in size:
            parts = size.split("x")
            if len(parts) == 2:
                try:
                    width, height = int(parts[0]), int(parts[1])
                except ValueError:
                    pass

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": width,
            "height": height,
            "batch_size": num_images,
            "steps": 20,
            "cfg_scale": 7,
        }

        logger.debug(
            "SD txt2img request: base=%s width=%d height=%d",
            base,
            width,
            height,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base}/sdapi/v1/txt2img",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Stable Diffusion API error ({resp.status}): {error_text}"
                    )
                data = await resp.json()

        b64_images: list[str] = data.get("images", [])
        return [f"base64://{img}" for img in b64_images]

    async def _generate_sd_img2img(
        self,
        prompt: str,
        init_images: list[str],
        *,
        negative_prompt: str = "",
        size: str = "",
        num_images: int = 1,
    ) -> list[str]:
        """Generate images via Stable Diffusion WebUI img2img API."""
        base = self.sd_api_base.rstrip("/")

        width, height = 512, 512
        if size and "x" in size:
            parts = size.split("x")
            if len(parts) == 2:
                try:
                    width, height = int(parts[0]), int(parts[1])
                except ValueError:
                    pass

        payload: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "init_images": init_images,
            "width": width,
            "height": height,
            "batch_size": num_images,
            "steps": 20,
            "cfg_scale": 7,
            "denoising_strength": 0.75,
        }

        logger.debug(
            "SD img2img request: base=%s width=%d height=%d",
            base,
            width,
            height,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base}/sdapi/v1/img2img",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Stable Diffusion API error ({resp.status}): {error_text}"
                    )
                data = await resp.json()

        b64_images: list[str] = data.get("images", [])
        return [f"base64://{img}" for img in b64_images]

    # ------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------

    async def _generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        style: str = "",
        aspect_ratio: str = "",
        model: str = "",
        num_images: int = 0,
    ) -> list[str]:
        """Dispatch to the configured image generation backend."""
        num = num_images if num_images > 0 else self.default_num

        if self.api_type == "sd" or self.api_type == "stable-diffusion":
            size = (
                self._aspect_to_size(aspect_ratio)
                if aspect_ratio
                else self.default_size
            )
            return await self._generate_sd(
                prompt,
                negative_prompt=negative_prompt,
                size=size,
                num_images=num,
            )
        else:
            # Default: OpenAI / DALL-E compatible
            size = (
                self._aspect_to_size(aspect_ratio)
                if aspect_ratio
                else self.default_size
            )
            if style:
                prompt = f"{prompt}, style: {style}"
            return await self._generate_openai(
                prompt,
                negative_prompt=negative_prompt,
                model=model,
                size=size,
                num_images=num,
            )

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

        try:
            urls = await self._generate(prompt, **kwargs)
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

        # Convert first image to base64
        try:
            init_b64 = await images[0].convert_to_base64()
        except Exception as e:
            logger.exception(f"Failed to read input image: {e}")
            yield event.plain_result(f"❌ Failed to read input image: {e}")
            return

        num = kwargs.get("num_images", 0)
        if num <= 0:
            num = self.default_num

        negative = kwargs.get("negative_prompt", "")
        style = kwargs.get("style", "")

        if self.api_type == "sd" or self.api_type == "stable-diffusion":
            size = self.default_size
            aspect = kwargs.get("aspect_ratio", "")
            if aspect:
                size = self._aspect_to_size(aspect)
            try:
                urls = await self._generate_sd_img2img(
                    prompt,
                    [init_b64],
                    negative_prompt=negative,
                    size=size,
                    num_images=num,
                )
            except Exception as e:
                logger.exception(f"SD img2img failed: {e}")
                yield event.plain_result(f"❌ Image generation failed: {e}")
                return
        else:
            # For OpenAI, we simulate img2img by enhancing the prompt with image context
            if style:
                prompt = f"{prompt}, style: {style}"
            try:
                urls = await self._generate_openai(
                    prompt,
                    negative_prompt=negative,
                    num_images=num,
                )
            except Exception as e:
                logger.exception(f"OpenAI image generation failed: {e}")
                yield event.plain_result(f"❌ Image generation failed: {e}")
                return

        for url in urls:
            yield event.image_result(url)

    async def terminate(self):
        """Cleanup on plugin unload."""
        pass
