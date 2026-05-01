"""Gen Image 插件。

支持命令:
- :draw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>]
- :pdraw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>] (需附带图片)

-p 和 -pre 至少提供一个。
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
        parser.add_argument("-p", "--prompt", default="", help="图像提示词")
        parser.add_argument("-pre", "--preset", default="", help="预设提示词名称")
        parser.add_argument(
            "-s",
            "--size",
            default="2048x2048",
            choices=GenImagePlugin.SIZES,
            help="图像尺寸",
        )
        parser.add_argument(
            "-q",
            "--quality",
            default="medium",
            choices=GenImagePlugin.QUALITIES,
            help="图像质量",
        )
        parser.add_argument(
            "-b",
            "--background",
            default="auto",
            choices=GenImagePlugin.BACKGROUNDS,
            help="背景模式",
        )
        return parser

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

    def _parse_named_args(self, raw_args: str) -> argparse.Namespace | None:
        """解析命令行风格参数。"""
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
        """调用 OpenAI 兼容 API 生成图像。"""
        if not self.api_key:
            raise ValueError("API 密钥未配置，请在插件设置中填写 api_key。")

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
                    raise RuntimeError(f"API 请求失败 ({resp.status}): {error_text}")
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
        if args is None:
            yield event.plain_result(
                "用法: :draw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>]\n"
                "-p 和 -pre 至少提供一个。"
            )
            return

        prompt = self._resolve_prompt(args.preset, args.prompt)
        if not prompt:
            if args.preset:
                yield event.plain_result(f"❌ 未知预设名: {args.preset}")
            else:
                yield event.plain_result(
                    "用法: :draw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>]\n"
                    "-p 和 -pre 至少提供一个。"
                )
            return

        yield event.plain_result("🎨 正在生成图像，请稍候...")

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
            logger.exception(f"图像生成失败: {e}")
            yield event.plain_result(f"❌ 图像生成失败: {e}")
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
        if args is None:
            yield event.plain_result(
                "用法: :pdraw -p <提示词> [-pre <预设名>] [-s <尺寸>] [-q <质量>] [-b <背景>] (需附带图片)\n"
                "-p 和 -pre 至少提供一个。"
            )
            return

        prompt = self._resolve_prompt(args.preset, args.prompt)
        if not prompt:
            if args.preset:
                yield event.plain_result(f"❌ 未知预设名: {args.preset}")
            else:
                yield event.plain_result("-p 和 -pre 至少提供一个。")
            return

        yield event.plain_result("🎨 正在处理图生图，请稍候...")

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
            logger.exception(f"图像生成失败: {e}")
            yield event.plain_result(f"❌ 图像生成失败: {e}")
            return

        for url in urls:
            yield event.image_result(url)

    async def terminate(self):
        """插件卸载时清理。"""
        pass
