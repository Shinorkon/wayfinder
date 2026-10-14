"""
Wayfinder SAM 3 (Segment Anything Model 3) Integration.
Reuses local weights from LightCraft (~/.config/lightcraft/models/sam3)
for zero-shot text-prompted instance segmentation on bus stop video frames.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import cv2
from transformers import Sam3Model, CLIPTokenizer


class SAM3Masker:
    """Zero-shot promptable semantic segmentation using Meta's SAM 3."""

    _instance: Optional["SAM3Masker"] = None

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True
    ):
        if not model_dir:
            model_dir = os.environ.get(
                "LIGHTCRAFT_SAM3_DIR",
                os.path.expanduser("~/.config/lightcraft/models/sam3")
            )
        self.model_dir = model_dir
        self.device = device
        self.dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

        print(f"[SAM3] Initializing from {model_dir} on {device} ({self.dtype})...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_dir)
        self.model = Sam3Model.from_pretrained(
            model_dir,
            torch_dtype=self.dtype
        ).to(self.device).eval()
        print("[SAM3] Model ready.")

    @classmethod
    def get_instance(cls) -> "SAM3Masker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.inference_mode()
    def segment_frame(
        self,
        frame: np.ndarray,
        text_prompts: Optional[List[str]] = None,
        conf_threshold: float = 0.45,
        max_masks_per_prompt: int = 8,
        min_mask_area_frac: float = 0.0005,
        allow_low_confidence_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Segments objects matching text prompts in a BGR video frame.

        Default prompt is deliberately narrow: the useful signal for stop analysis is
        "waiting passenger". A shelter prompt tends to return large structural blobs
        that dominate the overlay without adding information.

        ``allow_low_confidence_fallback`` is off by default because emitting the top-3
        masks when nothing cleared the threshold manufactures detections on frames that
        contain none, which is exactly the failure mode that makes CV dashboards lie.
        """
        if text_prompts is None:
            text_prompts = ["waiting passenger"]

        # Convert BGR to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        orig_w, orig_h = pil_img.size

        # Preprocessing: resize to 1008x1008, normalize to [-1, 1]
        resized = pil_img.resize((1008, 1008), Image.BILINEAR)
        px = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        px = (px - 0.5) / 0.5
        px = px.to(self.device, dtype=self.dtype)

        prompt_results: List[Dict[str, Any]] = []

        for prompt in text_prompts:
            enc = self.tokenizer(prompt, padding="max_length", max_length=32, return_tensors="pt")
            input_ids = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device)

            out = self.model(
                pixel_values=px,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            presence_score = float(torch.sigmoid(out.presence_logits).item())
            pred_logits = out.pred_logits[0]  # [N]
            scores = (torch.sigmoid(pred_logits) * presence_score).cpu().numpy()
            pred_masks = out.pred_masks[0]  # [N, H_mask, W_mask]

            # Select high scoring masks.
            valid_indices = np.where(scores >= conf_threshold)[0]
            used_fallback = False
            if len(valid_indices) == 0 and allow_low_confidence_fallback:
                valid_indices = np.argsort(scores)[-3:]
                used_fallback = True

            masks_data = []
            for idx in valid_indices[:max_masks_per_prompt]:
                mask_tensor = pred_masks[idx].float()
                # Upscale mask back to original image size
                mask_prob = torch.sigmoid(mask_tensor).unsqueeze(0).unsqueeze(0)
                mask_up = torch.nn.functional.interpolate(
                    mask_prob,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False
                )[0, 0].cpu().numpy()

                binary_mask = (mask_up > 0.5).astype(np.uint8)
                area_frac = float(binary_mask.sum()) / float(orig_w * orig_h)
                if area_frac < min_mask_area_frac:
                    continue
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                simplified_polygons = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 80:  # Filter noise specs
                        # Simplify contour
                        epsilon = 0.005 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        pts = approx.reshape(-1, 2).tolist()
                        simplified_polygons.append(pts)

                if simplified_polygons:
                    masks_data.append({
                        "score": float(round(scores[idx], 3)),
                        "polygons": simplified_polygons,
                        "area_pixels": int(np.sum(binary_mask)),
                        "area_frac": round(area_frac, 5),
                    })

            prompt_results.append({
                "prompt": prompt,
                "presence_score": float(round(presence_score, 3)),
                "masks": masks_data,
                "count": len(masks_data),
                "used_low_confidence_fallback": used_fallback,
            })

        return {
            "image_size": [orig_w, orig_h],
            "results": prompt_results,
            "total_segments": sum(len(p["masks"]) for p in prompt_results),
            "used_low_confidence_fallback": any(
                p["used_low_confidence_fallback"] for p in prompt_results
            ),
        }

    def mask_polygons(self, segment_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns the raw polygons for client-side drawing, in normalised coordinates.

        The live monitor streams these to the browser so SAM 3 masks appear over the
        playing video without the server having to encode a JPEG per update.
        """
        w, h = segment_results.get("image_size", [1, 1])
        out: List[Dict[str, Any]] = []
        for prompt_data in segment_results.get("results", []):
            for mask in prompt_data.get("masks", []):
                polys = []
                for poly in mask.get("polygons", []):
                    polys.append(
                        [[round(pt[0] / w, 5), round(pt[1] / h, 5)] for pt in poly]
                    )
                out.append(
                    {
                        "prompt": prompt_data.get("prompt"),
                        "score": mask.get("score"),
                        "area_frac": mask.get("area_frac"),
                        "polygons": polys,
                    }
                )
        return out

    def render_overlay(
        self,
        frame: np.ndarray,
        segment_results: Dict[str, Any],
        alpha: float = 0.40
    ) -> np.ndarray:
        """Draws translucent segmentation masks plus an honest caption."""
        overlay = frame.copy()
        output = frame.copy()
        h, w = output.shape[:2]
        scale = max(0.5, min(2.0, w / 1280.0))

        colors = [
            (214, 69, 139),   # Magenta (BGR)
            (0, 229, 255),     # Cyan (BGR)
            (16, 185, 129),    # Emerald (BGR)
            (245, 158, 11),    # Amber (BGR)
        ]

        for p_idx, prompt_data in enumerate(segment_results.get("results", [])):
            color = colors[p_idx % len(colors)]
            for mask in prompt_data.get("masks", []):
                for poly in mask.get("polygons", []):
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(output, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Caption: name the prompt and the count, and say so when the result was weak.
        parts = []
        for p in segment_results.get("results", []):
            tag = p["prompt"]
            if p.get("used_low_confidence_fallback"):
                tag += " (below threshold)"
            parts.append(f"{tag}: {p['count']}")
        badge = "SAM 3 - " + " | ".join(parts) if parts else "SAM 3 - no prompts"
        if segment_results.get("used_low_confidence_fallback"):
            badge += "  [low confidence]"

        font_scale = 0.5 * scale
        pad = int(round(10 * scale))
        thickness = max(1, int(round(2 * scale)))
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(output, (pad, pad), (pad + bw + 18, pad + bh + 16), (20, 10, 30), -1)
        cv2.rectangle(output, (pad, pad), (pad + bw + 18, pad + bh + 16), (214, 69, 139), thickness)
        cv2.putText(
            output, badge, (pad + 9, pad + bh + 9),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 255), thickness, cv2.LINE_AA,
        )

        return output
