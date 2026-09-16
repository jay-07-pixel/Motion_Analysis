"""Start menu and video file picker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InputChoice:
    """Result of the start-menu selection.

    Attributes:
        mode: ``camera`` for live RealSense or ``video`` for an uploaded file.
        video_path: Path from the file picker when mode is ``video``.
    """

    mode: str
    video_path: Optional[str] = None


def show_input_menu() -> Optional[InputChoice]:
    """Show the Live Camera / Upload Video selection screen.

    Returns:
        The chosen input, or None if the user closed the window.

    Raises:
        RuntimeError: If Tkinter is not available.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is required for the input-selection screen and video "
            "file picker. Install a Python build that includes tkinter."
        ) from exc

    choice: dict[str, Optional[InputChoice]] = {"value": None}

    root = tk.Tk()
    root.title("2D Motion Analysis")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    container = tk.Frame(root, padx=32, pady=28)
    container.pack()

    tk.Label(
        container,
        text="Select input",
        font=("Segoe UI", 16, "bold"),
    ).pack(pady=(0, 8))
    tk.Label(
        container,
        text="Live RealSense D455F or an uploaded video file.\n"
        "Both modes use the same 2D pose and hands pipeline.",
        justify="center",
    ).pack(pady=(0, 18))

    def choose_camera() -> None:
        choice["value"] = InputChoice(mode="camera")
        root.destroy()

    def choose_video() -> None:
        path = filedialog.askopenfilename(
            parent=root,
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        choice["value"] = InputChoice(mode="video", video_path=path)
        root.destroy()

    tk.Button(
        container,
        text="Live Camera",
        width=22,
        height=2,
        command=choose_camera,
    ).pack(pady=6)
    tk.Button(
        container,
        text="Upload Video",
        width=22,
        height=2,
        command=choose_video,
    ).pack(pady=6)

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 3
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    return choice["value"]
