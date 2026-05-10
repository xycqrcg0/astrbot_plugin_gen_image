"""Gen Image 插件。

支持命令:
- :draw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>]
- :pdraw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>] (需附带图片)

提供商: openai, nano_banana, gemini (在配置中切换)

-p 和 -pre 至少提供一个。
"""

import re
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
        api_base = str(openai_cfg.get("api_base", "")).strip().rstrip("/")
        model = str(openai_cfg.get("model", "")).strip()
        img_model = str(openai_cfg.get("img_model", "")).strip()
        self.txt2img_url = f"{api_base}/{model}" if (api_base and model) else ""
        self.img2img_url = f"{api_base}/{img_model}" if (api_base and img_model) else ""
        self.default_size = (
            str(openai_cfg.get("default_size", "2048x2048")).strip() or "2048x2048"
        )
        self.default_quality = (
            str(openai_cfg.get("default_quality", "medium")).strip() or "medium"
        )
        self.default_background = (
            str(openai_cfg.get("default_background", "auto")).strip() or "auto"
        )
        self.default_num = int(openai_cfg.get("default_num", 1))
        self.timeout = int(openai_cfg.get("timeout", 300))

        self.provider = (
            str(openai_cfg.get("provider", "openai")).strip().lower() or "openai"
        )

        # Preset prompts: key=value per line
        presets_cfg = config.get("presets", {})
        preset_text = str(presets_cfg.get("preset_list", "")).strip()
        self.presets: dict[str, str] = {}
        if preset_text:
            for line in preset_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    logger.warning(f"预设行缺少 '='，已忽略: {line}")
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key in self.presets:
                    logger.warning(f"预设键重复，将覆盖: {key}")
                self.presets[key] = value.strip()

    # ------------------------------------------------------------
    # Argument parser helpers
    # ------------------------------------------------------------

    SIZES = {
        "1024x1024",
        "1024x1536",
        "1536x1024",
        "2048x2048",
        "2048x1152",
        "3840x2160",
        "2160x3840",
    }
    QUALITIES = {"low", "medium", "high"}
    BACKGROUNDS = {"transparent", "opaque", "auto"}

    def _resolve_prompt(self, preset_key: str, user_prompt: str) -> str | None:
        """根据预设名和用户提示词解析最终提示词。

        返回完整提示词，如果两者都未提供则返回 None。
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

    def _parse_named_args(self, raw_args: str) -> dict[str, str]:
        """Parse named arguments from raw string using regex.

        -p/--prompt captures multi-word text until the next flag or end.
        Other flags capture a single token.
        Returns a dict; check '_error' key for validation failures.
        """
        result: dict[str, str] = {}

        # -p / --prompt: multi-word until next flag or EOS (re.DOTALL to handle multiline prompts)
        m = re.search(
            r"(?:-p|--prompt)\s+(.+?)(?=\s+--?[a-zA-Z]|$)", raw_args, re.DOTALL
        )
        if m:
            result["prompt"] = m.group(1).strip()

        # Single-value flags
        for short, long in [
            ("pre", "preset"),
            ("s", "size"),
            ("q", "quality"),
            ("b", "background"),
        ]:
            m = re.search(rf"(?:-{short}|--{long})\s+(\S+)", raw_args)
            if m:
                result[long] = m.group(1)

        # Validate choices with helpful error messages
        if "size" in result and result["size"] not in self.SIZES:
            result["_error"] = (
                f"无效尺寸: {result['size']}\n" f"可选: {', '.join(sorted(self.SIZES))}"
            )
        elif "quality" in result and result["quality"] not in self.QUALITIES:
            result["_error"] = (
                f"无效质量: {result['quality']}\n"
                f"可选: {', '.join(sorted(self.QUALITIES))}"
            )
        elif "background" in result and result["background"] not in self.BACKGROUNDS:
            result["_error"] = (
                f"无效背景: {result['background']}\n"
                f"可选: {', '.join(sorted(self.BACKGROUNDS))}"
            )

        return result

    # ------------------------------------------------------------
    # Image generation backend
    # ------------------------------------------------------------

    _ASPECT_MAP = {
        "1024x1024": "1:1",
        "2048x2048": "1:1",
        "1024x1536": "2:3",
        "1536x1024": "3:2",
        "2048x1152": "16:9",
        "3840x2160": "16:9",
        "2160x3840": "9:16",
    }

    # Resolution tier mapped from user-facing pixel dimensions.
    _NB_RESOLUTION_MAP = {
        "1024x1024": "1K",
        "1024x1536": "1K",
        "1536x1024": "1K",
        "2048x2048": "2K",
        "2048x1152": "2K",
        "3840x2160": "4K",
        "2160x3840": "4K",
    }

    async def _generate(
        self,
        prompt: str,
        *,
        size: str = "",
        quality: str = "",
        background: str = "",
        num_images: int = 1,
        image_data: str = "",
    ) -> list[str]:
        """Route to the appropriate provider's generation method."""
        if self.provider == "gemini":
            return await self._generate_gemini(prompt, size=size, image_data=image_data)
        if self.provider == "nano_banana":
            return await self._generate_nano_banana(
                prompt, size=size, quality=quality, image_data=image_data
            )
        # openai / any OpenAI-compatible
        url = self.img2img_url if image_data else self.txt2img_url
        if not url:
            raise ValueError(
                "API 地址未配置，请在插件设置中填写 api_base 和 model/img_model。"
            )
        return await self._generate_openai(
            url,
            prompt,
            size=size,
            quality=quality,
            background=background,
            num_images=num_images,
            image_data=image_data,
        )

    async def _generate_nano_banana(
        self,
        prompt: str,
        *,
        size: str = "",
        quality: str = "",
        image_data: str = "",
    ) -> list[str]:
        """Call Nano Banana 2 API for image generation."""
        if not self.api_key:
            raise ValueError("API 密钥未配置。")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "prompt": prompt,
            "size": self._NB_RESOLUTION_MAP.get(size, "1K"),
            "aspect_ratio": self._ASPECT_MAP.get(size, "1:1"),
        }
        if image_data:
            payload["image"] = image_data

        url = self.img2img_url if image_data else self.txt2img_url

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=None, connect=30, sock_read=self.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Nano Banana API 请求失败 ({resp.status}): {error_text}"
                    )
                data = await resp.json()

        return list(data.get("images", []))

    async def _generate_gemini(
        self,
        prompt: str,
        *,
        size: str = "",
        image_data: str = "",
    ) -> list[str]:
        """Call Gemini API via jiekou.ai for image generation."""
        if not self.api_key:
            raise ValueError("API 密钥未配置。")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "prompt": prompt,
            "size": self._NB_RESOLUTION_MAP.get(size, "1K"),
            "aspect_ratio": self._ASPECT_MAP.get(size, "1:1"),
        }
        if image_data:
            payload["image_base64s"] = [image_data]

        url = self.img2img_url if image_data else self.txt2img_url

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=None, connect=30, sock_read=self.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Gemini API 请求失败 ({resp.status}): {error_text}"
                    )
                data = await resp.json()

        return list(data.get("image_urls", []))

    async def _generate_openai(
        self,
        url: str,
        prompt: str,
        *,
        size: str = "",
        quality: str = "",
        background: str = "",
        num_images: int = 1,
        image_data: str = "",
    ) -> list[str]:
        """调用 OpenAI 兼容 API 生成图像。"""
        if not self.api_key:
            raise ValueError("API 密钥未配置，请在插件设置中填写 api_key。")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n": num_images,
            "quality": quality or self.default_quality,
            "background": background or self.default_background,
            "moderation": "low",
            "output_format": "png",
        }
        resolved_size = size or self.default_size
        if resolved_size:
            payload["size"] = resolved_size
        if image_data:
            payload["image"] = image_data

        logger.debug(
            "图像请求: url=%s size=%s quality=%s",
            url,
            payload.get("size", ""),
            payload["quality"],
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=None, connect=30, sock_read=self.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"API 请求失败 ({resp.status}): {error_text}")
                data = await resp.json()

        urls: list[str] = []
        # Support both: {"images": ["url1", ...]} and {"data": [{"url": "..."}, ...]}
        images = data.get("images")
        if isinstance(images, list):
            urls.extend(str(u) for u in images if u)
        else:
            for item in data.get("data", []):
                img_url = item.get("url")
                if img_url:
                    urls.append(img_url)
                elif item.get("b64_json"):
                    urls.append("base64://" + item["b64_json"])
        return urls

    # ------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------

    def _strip_command_prefix(self, message_str: str, command: str) -> str:
        """去除命令前缀，返回剩余参数。"""
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
        """文生图。"""
        raw_args = self._strip_command_prefix(event.message_str, "draw")
        args = self._parse_named_args(raw_args)
        if "_error" in args:
            yield event.plain_result(f"❌ {args['_error']}")
            return
        if not args.get("prompt") and not args.get("preset"):
            yield event.plain_result(
                "用法: :draw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>]\n"
                "-p 和 -pre 至少提供一个。"
            )
            return

        prompt = self._resolve_prompt(args.get("preset", ""), args.get("prompt", ""))
        if not prompt:
            if args.get("preset"):
                available = ", ".join(sorted(self.presets)) if self.presets else "(无)"
                yield event.plain_result(
                    f"❌ 未知预设名: {args.get('preset')}\n" f"可用预设: {available}"
                )
            else:
                yield event.plain_result(
                    "用法: :draw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>]\n"
                    "-p 和 -pre 至少提供一个。"
                )
            return

        yield event.plain_result("🎨 正在生成图像，请稍候...")

        try:
            urls = await self._generate(
                prompt,
                size=args.get("size", ""),
                quality=args.get("quality", ""),
                background=args.get("background", ""),
                num_images=self.default_num,
            )
        except Exception as e:
            logger.exception("图像生成失败")
            err_msg = str(e) or type(e).__name__
            yield event.plain_result(f"❌ 图像生成失败: {err_msg}")
            return

        if not urls:
            yield event.plain_result("❌ 图像生成失败: API 返回了空结果。")
            return
        for url in urls:
            yield event.image_result(url)

    @filter.command("pdraw")
    async def pdraw_command(self, event: AstrMessageEvent):
        """图生图。"""
        images = [c for c in event.get_messages() if isinstance(c, Comp.Image)]
        if not images:
            yield event.plain_result(
                "用法: :pdraw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>] (需附带图片)\n"
                "-p 和 -pre 至少提供一个。"
            )
            return

        raw_args = self._strip_command_prefix(event.message_str, "pdraw")
        args = self._parse_named_args(raw_args)
        if "_error" in args:
            yield event.plain_result(f"❌ {args['_error']}")
            return
        if not args.get("prompt") and not args.get("preset"):
            yield event.plain_result(
                "用法: :pdraw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>] (需附带图片)\n"
                "-p 和 -pre 至少提供一个。"
            )
            return

        prompt = self._resolve_prompt(args.get("preset", ""), args.get("prompt", ""))
        if not prompt:
            if args.get("preset"):
                available = ", ".join(sorted(self.presets)) if self.presets else "(无)"
                yield event.plain_result(
                    f"❌ 未知预设名: {args.get('preset')}\n" f"可用预设: {available}"
                )
            else:
                yield event.plain_result("-p 和 -pre 至少提供一个。")
            return

        yield event.plain_result("🎨 正在处理图生图，请稍候...")

        # Encode first attached image
        try:
            image_data = await images[0].convert_to_base64()
        except Exception as e:
            logger.exception(f"读取图片失败: {e}")
            yield event.plain_result(f"❌ 读取图片失败: {e}")
            return

        try:
            urls = await self._generate(
                prompt,
                size=args.get("size", ""),
                quality=args.get("quality", ""),
                background=args.get("background", ""),
                num_images=self.default_num,
                image_data=image_data,
            )
        except Exception as e:
            logger.exception("图像生成失败")
            err_msg = str(e) or type(e).__name__
            yield event.plain_result(f"❌ 图像生成失败: {err_msg}")
            return

        if not urls:
            yield event.plain_result("❌ 图像生成失败: API 返回了空结果。")
            return
        for url in urls:
            yield event.image_result(url)

    async def terminate(self):
        """插件卸载时清理。"""
        pass
