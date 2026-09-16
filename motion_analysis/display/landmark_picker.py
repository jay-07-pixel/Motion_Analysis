"""Landmark selection screen for 2D motion analysis."""

from __future__ import annotations

from typing import Optional

from motion_analysis.motion.landmarks import (
    LandmarkTarget,
    default_landmark,
    selectable_landmarks,
)


def show_landmark_picker(
    initial: Optional[LandmarkTarget] = None,
) -> Optional[LandmarkTarget]:
    """Let the user choose which body or hand landmark to analyse.

    Args:
        initial: Pre-selected catalog entry. Defaults to LEFT_WRIST.

    Returns:
        The chosen landmark, or None if the user closed the window.

    Raises:
        RuntimeError: If Tkinter is not available.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is required for the landmark-selection screen. "
            "Install a Python build that includes tkinter."
        ) from exc

    catalog = selectable_landmarks()
    selected = initial or default_landmark()
    choice: dict[str, Optional[LandmarkTarget]] = {"value": None}

    root = tk.Tk()
    root.title("Select landmark")
    root.resizable(True, True)
    root.attributes("-topmost", True)

    container = tk.Frame(root, padx=24, pady=20)
    container.pack(fill="both", expand=True)

    tk.Label(
        container,
        text="Select a landmark to analyse",
        font=("Segoe UI", 14, "bold"),
    ).pack(anchor="w")
    tk.Label(
        container,
        text="Measurements use that joint's smoothed 2D pixel coordinates.\n"
        "You can also press [ and ] in the video window to change it.",
        justify="left",
    ).pack(anchor="w", pady=(4, 12))

    columns = ("group", "name")
    tree = ttk.Treeview(
        container,
        columns=columns,
        show="headings",
        height=16,
        selectmode="browse",
    )
    tree.heading("group", text="Group")
    tree.heading("name", text="Landmark")
    tree.column("group", width=120, stretch=False)
    tree.column("name", width=220, stretch=True)

    scrollbar = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    list_frame = tk.Frame(container)
    list_frame.pack(fill="both", expand=True)
    tree.pack(side="left", fill="both", expand=True, in_=list_frame)
    scrollbar.pack(side="right", fill="y", in_=list_frame)

    item_keys: dict[str, LandmarkTarget] = {}
    initial_item = None
    for target in catalog:
        item_id = tree.insert("", "end", values=(target.group, target.name))
        item_keys[item_id] = target
        if target.key == selected.key:
            initial_item = item_id

    if initial_item is not None:
        tree.selection_set(initial_item)
        tree.see(initial_item)
        tree.focus(initial_item)

    def current_target() -> Optional[LandmarkTarget]:
        focused = tree.focus()
        if focused and focused in item_keys:
            return item_keys[focused]
        selection = tree.selection()
        if selection and selection[0] in item_keys:
            return item_keys[selection[0]]
        return selected

    def confirm() -> None:
        choice["value"] = current_target()
        root.destroy()

    def cancel() -> None:
        choice["value"] = None
        root.destroy()

    tree.bind("<Double-1>", lambda _event: confirm())
    tree.bind("<Return>", lambda _event: confirm())

    buttons = tk.Frame(container)
    buttons.pack(fill="x", pady=(14, 0))
    tk.Button(buttons, text="Cancel", width=12, command=cancel).pack(side="right")
    tk.Button(buttons, text="Analyse", width=12, command=confirm).pack(
        side="right", padx=(0, 8)
    )

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.update_idletasks()
    width = max(root.winfo_width(), 420)
    height = max(root.winfo_height(), 460)
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 4
    root.geometry(f"{width}x{height}+{x}+{y}")
    tree.focus_set()
    root.mainloop()
    return choice["value"]
