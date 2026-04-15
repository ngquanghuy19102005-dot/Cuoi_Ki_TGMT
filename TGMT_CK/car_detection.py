import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from collections import defaultdict

# ============================================================
# CẤU HÌNH CỐ ĐỊNH
# ============================================================
MODEL_PATH     = "yolov8n.pt"
CONF_THRESHOLD = 0.4

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

COLORS = {
    1: (255, 255,   0),   # Vàng     - Bicycle
    2: (  0, 255,   0),   # Xanh lá  - Car
    3: (255, 165,   0),   # Cam      - Motorcycle
    5: (  0,   0, 255),   # Đỏ       - Bus
    7: (255,   0, 255),   # Tím      - Truck
}

# ============================================================
# CHỌN FILE VIDEO QUA HỘP THOẠI
# ============================================================
def chon_video():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Chon file video can xu ly",
        filetypes=[
            ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("Tat ca",  "*.*"),
        ],
    )
    root.destroy()

    if not path:
        messagebox.showwarning("Thong bao", "Khong chon file. Chuong trinh se thoat.")
        return None, None

    folder  = os.path.dirname(path)
    stem    = os.path.splitext(os.path.basename(path))[0]
    out     = os.path.join(folder, f"{stem}_output.mp4")
    return path, out


# ============================================================
# CLASS THEO DÕI VÀ PHÂN LOẠI XE
# ============================================================
class VehicleTracker:
    """
    Theo dõi các phương tiện trong frame, phân loại theo loại xe.
    Không dùng counting line — chỉ nhận diện và phân loại.
    """

    def __init__(self):
        # cls_history[tid] = {cls_name: count}  → lấy class chiếm ưu thế
        self.cls_history: dict[int, dict[str, int]] = {}

        # Tổng số xe theo loại (xuất hiện trong video, không trùng)
        self.seen_ids:     set[int]         = set()
        self.id_to_cls:    dict[int, str]   = {}
        self.total_by_cls: dict[str, int]   = defaultdict(int)

        # flash[tid] = số frame còn highlight (khi xe mới xuất hiện)
        self.flash: dict[int, int] = {}

    # ----------------------------------------------------------
    def _dominant_cls(self, tid: int) -> str:
        hist = self.cls_history.get(tid, {})
        if not hist:
            return "Car"
        return max(hist, key=hist.get)

    # ----------------------------------------------------------
    def update(self, tid: int, cls_name: str) -> bool:
        """
        Cập nhật lịch sử class cho track.
        Trả về True nếu đây là lần đầu tiên thấy track này.
        """
        hist = self.cls_history.setdefault(tid, {})
        hist[cls_name] = hist.get(cls_name, 0) + 1

        is_new = tid not in self.seen_ids
        if is_new:
            self.seen_ids.add(tid)
            self.flash[tid] = 15   # highlight 15 frame khi xe mới xuất hiện

        # Cập nhật thống kê tổng (theo dominant class hiện tại)
        dominant = self._dominant_cls(tid)
        if tid in self.id_to_cls:
            old_cls = self.id_to_cls[tid]
            if old_cls != dominant:
                # Class thay đổi → cập nhật lại thống kê
                self.total_by_cls[old_cls] = max(0, self.total_by_cls[old_cls] - 1)
                self.total_by_cls[dominant] += 1
                self.id_to_cls[tid] = dominant
        else:
            self.total_by_cls[dominant] += 1
            self.id_to_cls[tid] = dominant

        return is_new

    # ----------------------------------------------------------
    def cleanup(self, active_ids: set):
        """Xóa track đã ra khỏi frame để tránh rò rỉ bộ nhớ."""
        gone = set(self.cls_history.keys()) - active_ids
        for tid in gone:
            self.cls_history.pop(tid, None)
            self.flash.pop(tid, None)
            # Giữ lại seen_ids, id_to_cls, total_by_cls để thống kê chính xác

    # ----------------------------------------------------------
    def tick_flash(self, tid: int) -> bool:
        if self.flash.get(tid, 0) > 0:
            self.flash[tid] -= 1
            return True
        return False

    # ----------------------------------------------------------
    @property
    def active_count(self) -> int:
        """Số xe đang trong frame (đã thấy tổng cộng)."""
        return len(self.seen_ids)

    @property
    def grand_total(self) -> int:
        return sum(self.total_by_cls.values())


# ============================================================
# VẼ GIAO DIỆN TRÊN FRAME
# ============================================================
def ve_giao_dien(frame, tracker, W, H, frame_no, total_frames, active_in_frame):
    """
    Vẽ HUD thống kê phương tiện lên frame.
    - Góc trên trái: thống kê theo từng loại xe
    - Góc trên phải: tổng xe & số xe trong frame
    - Góc dưới: thanh tiến trình (nếu cần)
    """

    # ── Bảng thống kê theo loại xe (trái) ──────────────────────
    panel_x, panel_y = 10, 10
    row_h    = 26
    n_cls    = len(VEHICLE_CLASSES)
    panel_h  = 36 + n_cls * row_h
    panel_w  = 210

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "PHAN LOAI XE",
                (panel_x + 8, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    for i, (cid, cname) in enumerate(VEHICLE_CLASSES.items()):
        cnt   = tracker.total_by_cls.get(cname, 0)
        color = COLORS[cid]
        ry    = panel_y + 36 + i * row_h

        # Chấm màu loại xe
        cv2.circle(frame, (panel_x + 14, ry - 4), 5, color, -1)

        # Tên + số lượng
        cv2.putText(frame,
                    f"{cname:<12}: {cnt:>4}",
                    (panel_x + 26, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)

        # Bar chart mini
        bar_len = min(cnt * 3, 70)
        cv2.rectangle(frame,
                      (panel_x + 165, ry - 10),
                      (panel_x + 165 + bar_len, ry - 2),
                      color, -1)

    # ── Góc trên phải: tổng & số xe hiện tại ──────────────────
    total_lbl   = f"TONG: {tracker.grand_total}"
    active_lbl  = f"TRONG FRAME: {active_in_frame}"

    (tw, _), _ = cv2.getTextSize(total_lbl,  cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    (aw, _), _ = cv2.getTextSize(active_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)

    bw = max(tw, aw) + 24
    bh = 72
    bx = W - bw - 10
    by = 10

    ov2 = frame.copy()
    cv2.rectangle(ov2, (bx, by), (bx + bw, by + bh), (15, 15, 15), -1)
    cv2.addWeighted(ov2, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, total_lbl,
                (bx + 12, by + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 180), 2)
    cv2.putText(frame, active_lbl,
                (bx + 12, by + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1)

    # ── Frame counter (góc dưới trái nhỏ) ─────────────────────
    if total_frames > 0:
        cv2.putText(frame,
                    f"Frame {frame_no}/{total_frames}",
                    (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)


# ============================================================
# HÀM XỬ LÝ CHÍNH
# ============================================================
def xu_ly_video(video_path: str, output_path: str):
    # --- Load model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Khong tim thay {MODEL_PATH}. Dang tai tu internet...")
    model = YOLO(MODEL_PATH)
    print("Model san sang!")

    # --- Mở video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H} | {fps:.1f} FPS | {total} frames")

    tracker  = VehicleTracker()

    # --- Chuẩn bị ghi output ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        print(f"\rDang xu ly: {frame_no}/{total}  "
              f"Tong={tracker.grand_total}",
              end="", flush=True)

        # --- Detect + Track ---
        results = model.track(
            frame, persist=True, verbose=False,
            tracker="bytetrack.yaml"
        )[0]

        active_ids: set[int] = set()
        active_in_frame      = 0

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            tids  = results.boxes.id.cpu().numpy().astype(int)
            clss  = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, tid, cid, conf in zip(boxes, tids, clss, confs):
                if cid not in VEHICLE_CLASSES:
                    continue
                if conf < CONF_THRESHOLD:
                    continue

                cname = VEHICLE_CLASSES[cid]
                color = COLORS[cid]
                active_ids.add(tid)
                active_in_frame += 1

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                is_new = tracker.update(tid, cname)

                # ── Vẽ bounding box ──────────────────────────
                thickness = 3 if (is_new or tracker.tick_flash(tid)) else 2
                box_color = (0, 255, 255) if (is_new or tracker.flash.get(tid, 0) > 0) else color

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                # ── Nhãn phía trên box ───────────────────────
                dominant = tracker._dominant_cls(tid)
                lbl = f"[{tid}] {dominant} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              box_color, -1)
                cv2.putText(frame, lbl, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

                # ── Chấm tâm ─────────────────────────────────
                cv2.circle(frame, (cx, cy), 4, box_color, -1)

        tracker.cleanup(active_ids)

        # Vẽ HUD
        ve_giao_dien(frame, tracker, W, H, frame_no, total, active_in_frame)

        cv2.imshow("Nhan dien phuong tien giao thong  [Q = thoat]", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nDa dung som.")
            break

    # --- Dọn dẹp ---
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nDa luu: {output_path}")

    # --- Bảng tổng kết ---
    print("\n" + "=" * 50)
    print("  KET QUA CUOI CUNG")
    print("=" * 50)
    print(f"  Tong so phuong tien phat hien: {tracker.grand_total}")
    print()
    for cid, cname in VEHICLE_CLASSES.items():
        cnt = tracker.total_by_cls.get(cname, 0)
        if cnt == 0:
            continue
        bar = "█" * min(cnt, 30) + (f" +{cnt - 30}" if cnt > 30 else "")
        print(f"  {cname:<12}: {cnt:>4}  {bar}")
    print("=" * 50)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    video_in, video_out = chon_video()
    if video_in:
        print(f"Input : {video_in}")
        print(f"Output: {video_out}")
        xu_ly_video(video_in, video_out)