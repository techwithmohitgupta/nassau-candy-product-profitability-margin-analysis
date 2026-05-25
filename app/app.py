import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image, ImageChops

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Product Line Profitability & Margin Performance Analysis",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="auto",
)

# =========================================================
# PATH CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR.parent / "data" / "processed"

CSS_FILE = ASSETS_DIR / "nc_dashboard.css"
NASSAU_LOGO = ASSETS_DIR / "nassau_candy.png"
NASSAU_LOGO_TRIMMED = ASSETS_DIR / "nassau_candy_trimmed.png"
UNIFIED_LOGO = ASSETS_DIR / "unified_mentor.png"

DISPLAY_NASSAU_LOGO = (
    NASSAU_LOGO_TRIMMED if NASSAU_LOGO_TRIMMED.exists() else NASSAU_LOGO
)

# =========================================================
# DASHBOARD DATA FILE REGISTRY
# =========================================================
DATA_FILES = {
    "bottom_products": "bottom_products.csv",
    "cleaned": "cleaned_nassau_candy.csv",
    "cost_sales_scatter": "cost_sales_scatter_dataset.csv",
    "cost_vs_profit": "cost_vs_profit_analysis.csv",
    "kpi_summary": "dashboard_kpi_summary.csv",
    "dependency": "dependency_indicators.csv",
    "division_level": "division_level_analysis.csv",
    "feature_engineered": "feature_engineered_nassau_candy.csv",
    "final_feature": "final_feature_engineered_dataset.csv",
    "margin_risk": "margin_risk_products.csv",
    "pareto_profit": "pareto_profit_analysis.csv",
    "pareto_sales": "pareto_sales_analysis.csv",
    "product_level": "product_level_analysis.csv",
    "product_margin_leaderboard": "product_margin_leaderboard.csv",
    "segment_performance": "segment_performance.csv",
    "top_profit": "top_profit_products.csv",
    "top_sales": "top_sales_products.csv",
}

# =========================================================
# LOAD CSS
# =========================================================
def load_css(css_path: Path) -> None:
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css(CSS_FILE)

# =========================================================
# LOGO IMAGE TRIM HELPER
# Fixes visual off-center logo caused by extra whitespace in PNG canvas.
# =========================================================
@st.cache_data(show_spinner=False)
def get_trimmed_logo_path(logo_path: str) -> str:
    source_path = Path(logo_path)

    if not source_path.exists():
        return str(source_path)

    output_path = source_path.with_name(f"{source_path.stem}_trimmed.png")

    if output_path.exists():
        return str(output_path)

    image = Image.open(source_path).convert("RGBA")

    # Trim transparent whitespace first
    alpha = image.split()[-1]
    bbox = alpha.getbbox()

    if bbox:
        cropped = image.crop(bbox)
    else:
        # Fallback for non-transparent white/cream background images
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        bbox = diff.getbbox()
        cropped = image.crop(bbox) if bbox else image

    # Add balanced padding after crop
    pad_x = 28
    pad_y = 14
    final_img = Image.new(
        "RGBA",
        (cropped.width + pad_x * 2, cropped.height + pad_y * 2),
        (255, 255, 255, 0),
    )
    final_img.paste(cropped, (pad_x, pad_y), cropped)

    final_img.save(output_path)
    return str(output_path)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def clean_key(value) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("%", "percent")
        .replace("/", " ")
        .replace("(", "")
        .replace(")", "")
    )


def find_col(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    col_map = {clean_key(col): col for col in df.columns}

    for name in possible_names:
        key = clean_key(name)
        if key in col_map:
            return col_map[key]

    for col in df.columns:
        col_clean = clean_key(col)
        for name in possible_names:
            name_clean = clean_key(name)
            if name_clean in col_clean or col_clean in name_clean:
                return col

    return None


def to_num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt_money(value, decimals=0) -> str:
    try:
        return f"${float(value):,.{decimals}f}"
    except Exception:
        return "$0"


def fmt_money_compact(value) -> str:
    """Executive chart label format: $19.4K instead of long labels inside visuals."""
    try:
        numeric_value = float(value)
    except Exception:
        return "$0"

    sign = "-" if numeric_value < 0 else ""
    numeric_value = abs(numeric_value)

    if numeric_value >= 1_000_000:
        return f"{sign}${numeric_value / 1_000_000:.1f}M"
    if numeric_value >= 1_000:
        return f"{sign}${numeric_value / 1_000:.1f}K"
    return f"{sign}${numeric_value:,.0f}"


def fmt_pct(value, decimals=1) -> str:
    try:
        return f"{float(value):,.{decimals}f}%"
    except Exception:
        return "0.0%"


def fmt_num(value, decimals=2) -> str:
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "0.00"



# =========================================================
# STEP 7C — TABLE FORMATTING POLISH HELPERS
# Safe mode: display formatting only. Raw calculation data stays unchanged.
# =========================================================
def fmt_table_money(value, decimals=0) -> str:
    try:
        if pd.isna(value):
            return "$0"
        return f"${float(value):,.{decimals}f}"
    except Exception:
        return "$0"


def fmt_table_pct(value, decimals=1) -> str:
    try:
        if pd.isna(value):
            return "0.0%"
        return f"{float(value):,.{decimals}f}%"
    except Exception:
        return "0.0%"


def fmt_table_int(value) -> str:
    try:
        if pd.isna(value):
            return "0"
        return f"{int(float(value)):,}"
    except Exception:
        return "0"


def build_display_table(
    table: pd.DataFrame,
    money_columns: list[str] | None = None,
    money_2_columns: list[str] | None = None,
    pct_columns: list[str] | None = None,
    int_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Format table values for executive readability without changing source logic."""
    display_table = table.copy()

    money_columns = money_columns or []
    money_2_columns = money_2_columns or []
    pct_columns = pct_columns or []
    int_columns = int_columns or []

    for column in money_columns:
        if column in display_table.columns:
            display_table[column] = display_table[column].apply(lambda value: fmt_table_money(value, 0))

    for column in money_2_columns:
        if column in display_table.columns:
            display_table[column] = display_table[column].apply(lambda value: fmt_table_money(value, 2))

    for column in pct_columns:
        if column in display_table.columns:
            display_table[column] = display_table[column].apply(lambda value: fmt_table_pct(value, 1))

    for column in int_columns:
        if column in display_table.columns:
            display_table[column] = display_table[column].apply(fmt_table_int)

    return display_table


def get_table_column_config(table: pd.DataFrame) -> dict:
    """Stable column widths for cleaner Streamlit tables after display formatting."""
    column_config = {}

    width_map = {
        "Product": "large",
        "Division": "medium",
        "Rank": "small",
        "Sales": "small",
        "Revenue": "small",
        "Cost": "small",
        "Gross Profit": "small",
        "Gross Margin (%)": "small",
        "Cost-to-Sales (%)": "small",
        "Profit per Unit": "small",
        "Revenue Contribution (%)": "medium",
        "Profit Contribution (%)": "medium",
        "Cumulative Profit (%)": "medium",
        "Risk Flag": "medium",
        "Margin Status": "medium",
        "Pareto Tier": "medium",
        "Dependency Status": "medium",
    }

    for column in table.columns:
        column_config[column] = st.column_config.TextColumn(
            column,
            width=width_map.get(column, "medium"),
        )

    return column_config

def short_label(value, max_chars=28) -> str:
    text = str(value)

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3] + "..."


# =========================================================
# STEP 6B — CHART LABEL, LEGEND & TOOLTIP REFINEMENT
# Safe-mode helpers for cleaner chart labels while keeping full names in hover.
# =========================================================
def compact_product_label(value, max_chars=24) -> str:
    """Shorter product label for axes/legends; full name stays in hover."""
    text = str(value).strip()
    replacements = {
        "Wonka Bar - ": "Wonka - ",
        "Scrumdiddlyumptious": "Scrumdid.",
        "Triple Dazzle Caramel": "Triple Dazzle",
        "Nutty Crunch Surprise": "Nutty Crunch",
        "Fudge Mallows": "Fudge Mallows",
        "Everlasting Gobstopper": "Everlasting Gob.",
        "Fizzy Lifting Drinks": "Fizzy Lifting",
        "Lickable Wallpaper": "Lickable Wallpaper",
    }

    for old_text, new_text in replacements.items():
        text = text.replace(old_text, new_text)

    return short_label(text, max_chars)


def compact_legend_label(value, max_chars=20) -> str:
    """Compact legend label to avoid bottom legend clipping."""
    if str(value) == "Remaining Portfolio":
        return "Remaining Portfolio"
    if str(value) == "Other Products":
        return "Other Products"
    return compact_product_label(value, max_chars)


def pareto_axis_label(value, max_chars=18) -> str:
    """Compact Pareto labels with controlled wrapping for dense x-axis."""
    label = compact_product_label(value, max_chars)
    if " - " in label:
        label = label.replace(" - ", "<br>", 1)
    elif " " in label and len(label) > 12:
        parts = label.split(" ")
        midpoint = max(1, len(parts) // 2)
        label = " ".join(parts[:midpoint]) + "<br>" + " ".join(parts[midpoint:])
    return label


# =========================================================
# STEP 6C — FINAL VISUALIZATION MICRO-FIXES
# Safe-mode helpers for final chart polish.
# =========================================================
def get_density_chart_height(row_count: int, base_height: int = 440, single_height: int = 360) -> int:
    """Reduce empty visual weight for single/low-density chart states."""
    try:
        row_count = int(row_count)
    except Exception:
        row_count = 0

    if row_count <= 1:
        return single_height
    if row_count <= 3:
        return max(single_height + 30, base_height - 34)
    return base_height


def add_single_point_annotation(fig, data_frame: pd.DataFrame, x_col: str, y_col: str, label_col: str):
    """Add one safe in-chart label for single-point scatter cases without cutting at edges."""
    if data_frame is None or data_frame.empty or len(data_frame) != 1:
        return fig

    row = data_frame.iloc[0]
    x_value = float(row.get(x_col, 0) or 0)
    y_value = float(row.get(y_col, 0) or 0)
    max_x_value = float(data_frame[x_col].max() or 0)

    x_shift = -54 if max_x_value and x_value >= max_x_value * 0.72 else 54
    x_anchor = "right" if x_shift < 0 else "left"

    fig.add_annotation(
        x=x_value,
        y=y_value,
        text=compact_product_label(row.get(label_col, ""), 18),
        showarrow=False,
        xshift=x_shift,
        yshift=18,
        xanchor=x_anchor,
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(36,50,74,0.12)",
        borderwidth=1,
        borderpad=4,
        font=dict(size=10, color=NC_COLORS["text"]),
    )
    return fig


# =========================================================
# STEP 6A — GLOBAL PLOTLY THEME SYSTEM
# Dynamic, chart-aware Nassau Candy visual language.
# =========================================================
NC_COLORS = {
    "navy": "#24324a",
    "navy_2": "#31425f",
    "teal": "#0f8b72",
    "teal_2": "#4f8f8a",
    "mint": "#8bd8cf",
    "mint_2": "#79c7b8",
    "coral": "#f48770",
    "coral_2": "#ff806d",
    "beige": "#f1b993",
    "sand": "#b9895f",
    "slate": "#6b7c93",
    "soft": "#d7f1ed",
    "grid": "rgba(36, 50, 74, 0.105)",
    "zero": "rgba(36, 50, 74, 0.18)",
    "axis": "#64748b",
    "text": "#24324a",
    "card": "rgba(255,255,255,0.0)",
}

NASSAU_PRODUCT_PALETTE = [
    "#0f8b72",  # executive teal
    "#24324a",  # deep navy
    "#4f8f8a",  # muted teal
    "#d98f76",  # warm coral-brown
    "#b9895f",  # caramel
    "#79c7b8",  # soft teal
    "#31425f",  # slate navy
    "#f1b993",  # beige
    "#8bd8cf",  # mint
    "#6b7c93",  # slate
    "#f48770",  # coral
    "#557c83",  # blue teal
    "#a66f58",  # cocoa
    "#91b8aa",  # sage
    "#c69272",  # nougat
    "#1f5f68",  # deep aqua
    "#e7a08c",  # soft coral
    "#49627c",  # blue slate
]

RISK_COLOR_MAP = {
    "Stable": NC_COLORS["teal"],
    "Healthy Margin": NC_COLORS["teal"],
    "Margin Risk": NC_COLORS["coral"],
    "Below Threshold": NC_COLORS["coral"],
    "High Cost Load": NC_COLORS["beige"],
    "Profit Risk": "#b91c1c",
    "Portfolio Mix": NC_COLORS["mint_2"],
}

DEPENDENCY_COLOR_MAP = {
    "Stable": NC_COLORS["teal"],
    "High": NC_COLORS["coral"],
    "High Dependency": NC_COLORS["coral"],
    "Moderate Dependency": NC_COLORS["beige"],
    "Balanced Portfolio": NC_COLORS["teal"],
    "Core Driver": NC_COLORS["navy"],
    "Distributed": NC_COLORS["mint"],
}


def stable_color_index(value: str, palette_size: int) -> int:
    """Deterministic index so a product keeps its color across filters and reruns."""
    text = str(value).strip().lower()
    if palette_size <= 0:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % palette_size


def build_product_color_map(product_names: list[str]) -> dict[str, str]:
    """Create stable product-to-color mapping independent of current filter size."""
    unique_products = [str(product) for product in product_names if pd.notna(product)]
    unique_products = list(dict.fromkeys(unique_products))

    product_color_map = {}
    for product in unique_products:
        product_color_map[product] = NASSAU_PRODUCT_PALETTE[
            stable_color_index(product, len(NASSAU_PRODUCT_PALETTE))
        ]

    return product_color_map


def apply_dynamic_product_colors(
    data_frame: pd.DataFrame,
    product_column: str = "product_name",
    status_column: str = "margin_status",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assign product-based colors with coral risk override for below-threshold products."""
    if data_frame is None or data_frame.empty or product_column not in data_frame.columns:
        return data_frame, {}

    color_map = build_product_color_map(
        data_frame[product_column].astype(str).sort_values().tolist()
    )

    if status_column in data_frame.columns:
        below_threshold_products = data_frame.loc[
            data_frame[status_column].astype(str).eq("Below Threshold"),
            product_column,
        ].astype(str)

        for product in below_threshold_products:
            color_map[product] = "#f48770"

    colored_df = data_frame.copy()
    colored_df["product_color"] = (
        colored_df[product_column].astype(str).map(color_map).fillna("#24324a")
    )

    return colored_df, color_map


def build_selected_vs_remaining_profit_share(
    selected_product_name: str,
    product_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
) -> pd.DataFrame:
    """For single-product selection, compare selected product profit vs remaining portfolio."""
    if (
        not selected_product_name
        or product_summary is None
        or product_summary.empty
        or portfolio_summary is None
        or portfolio_summary.empty
    ):
        return pd.DataFrame()

    selected_profit = product_summary["profit"].sum()
    selected_sales = product_summary["sales"].sum()
    selected_margin = (selected_profit / selected_sales * 100) if selected_sales else 0

    portfolio_profit = portfolio_summary["profit"].sum()
    portfolio_sales = portfolio_summary["sales"].sum()

    remaining_profit = max(portfolio_profit - selected_profit, 0)
    remaining_sales = max(portfolio_sales - selected_sales, 0)
    remaining_margin = (remaining_profit / remaining_sales * 100) if remaining_sales else 0

    comparison_df = pd.DataFrame(
        [
            {
                "product_name": selected_product_name,
                "product_short": short_label(selected_product_name, 24),
                "sales": selected_sales,
                "profit": selected_profit,
                "gross_margin_pct": selected_margin,
                "profit_contribution_pct": (
                    selected_profit / portfolio_profit * 100
                    if portfolio_profit
                    else 0
                ),
                "slice_type": "Selected Product",
            },
            {
                "product_name": "Remaining Portfolio",
                "product_short": "Remaining Portfolio",
                "sales": remaining_sales,
                "profit": remaining_profit,
                "gross_margin_pct": remaining_margin,
                "profit_contribution_pct": (
                    remaining_profit / portfolio_profit * 100
                    if portfolio_profit
                    else 0
                ),
                "slice_type": "Remaining Portfolio",
            },
        ]
    )

    return comparison_df[comparison_df["profit"] > 0].copy()


def is_single_product_mode(selected_product_name: str, product_summary: pd.DataFrame) -> bool:
    return selected_product_name != "All Products" and product_summary is not None and len(product_summary) == 1


def build_single_product_insight(product_summary: pd.DataFrame, threshold_value: float) -> str:
    if product_summary is None or product_summary.empty:
        return ""

    row = product_summary.iloc[0]
    product_name = str(row.get("product_name", "Selected product"))
    margin_pct = float(row.get("gross_margin_pct", 0))
    profit_value = float(row.get("profit", 0))
    profit_share = float(row.get("profit_contribution_pct", 0))
    status = str(row.get("margin_status", "Healthy Margin"))

    if status == "Below Threshold":
        return (
            f"Selected product insight: **{product_name}** is below the selected "
            f"{threshold_value:.0f}% margin benchmark with **{margin_pct:.1f}% gross margin**, "
            f"generating **${profit_value:,.0f} gross profit** and contributing "
            f"**{profit_share:.1f}%** of the current filtered profit."
        )

    return (
        f"Selected product insight: **{product_name}** is above the selected "
        f"{threshold_value:.0f}% margin benchmark with **{margin_pct:.1f}% gross margin**, "
        f"generating **${profit_value:,.0f} gross profit** and contributing "
        f"**{profit_share:.1f}%** of the current filtered profit."
    )


def get_product_chart_height(single_product_view: bool) -> int:
    return 360 if single_product_view else 440


def get_donut_text_template(single_product_view: bool) -> str:
    return "%{label}<br>%{percent}" if single_product_view else "%{percent}"


# =========================================================
# STEP 7D — DONUT CONTEXT + LEGEND POLISH HELPERS
# Safe mode: display/legend context only. Donut calculations remain unchanged.
# =========================================================
def get_donut_context_caption(single_product_view: bool) -> str:
    if single_product_view:
        return "Selected product vs remaining filtered portfolio."
    return ""


def build_donut_legend_label(product_name: str, single_product_view: bool) -> str:
    product_text = str(product_name)

    if single_product_view and product_text != "Remaining Portfolio":
        return f"Selected: {compact_product_label(product_text, 16)}"

    if product_text == "Remaining Portfolio":
        return "Remaining Portfolio"

    if product_text == "Other Products":
        return "Other Products"

    return compact_legend_label(product_text, 18)


def get_donut_center_text(single_product_view: bool) -> str:
    if single_product_view:
        return "Selected vs<br>Remaining"
    return "Filtered<br>Profit Share"


def apply_nc_plotly_theme(
    fig,
    height: int = 440,
    margin: dict | None = None,
    showlegend: bool | None = None,
    legend_y: float = -0.22,
    legend_orientation: str = "h",
    legend_x: float = 0.5,
):
    """Apply a consistent premium Plotly layout without changing chart data."""
    if margin is None:
        margin = dict(l=18, r=46, t=22, b=78)

    layout_updates = {
        "height": height,
        "margin": margin,
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color=NC_COLORS["text"], size=12, family="Inter, Segoe UI, Arial"),
        "hoverlabel": dict(
            bgcolor="white",
            bordercolor="rgba(36,50,74,0.16)",
            font_size=12,
            font_color=NC_COLORS["text"],
            namelength=-1,
        ),
        "uniformtext_minsize": 8,
        "uniformtext_mode": "hide",
    }

    if showlegend is not None:
        layout_updates["showlegend"] = showlegend

    if showlegend:
        layout_updates["legend"] = dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=legend_x,
            font=dict(size=10, color=NC_COLORS["text"]),
            title_text="",
            tracegroupgap=8,
            itemsizing="constant",
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        )

    fig.update_layout(**layout_updates)

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="rgba(36, 50, 74, 0.16)",
        gridcolor=NC_COLORS["grid"],
        zerolinecolor=NC_COLORS["zero"],
        tickfont=dict(size=10.5, color=NC_COLORS["axis"]),
        title_font=dict(size=11.5, color=NC_COLORS["axis"]),
        automargin=True,
    )
    fig.update_yaxes(
        showline=False,
        gridcolor=NC_COLORS["grid"],
        zerolinecolor=NC_COLORS["zero"],
        tickfont=dict(size=10.5, color=NC_COLORS["axis"]),
        title_font=dict(size=11.5, color=NC_COLORS["axis"]),
        automargin=True,
    )

    return fig


def product_status_color(product_name: str, status: str | None = None) -> str:
    """Product-aware dynamic color with margin-risk override."""
    if status == "Below Threshold":
        return NC_COLORS["coral"]
    return NASSAU_PRODUCT_PALETTE[
        stable_color_index(product_name, len(NASSAU_PRODUCT_PALETTE))
    ]
    
# =========================================================
# GLOBAL PLOTLY RENDER CONFIG
# Safe mode: improves chart performance/readability without changing chart logic.
# =========================================================
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_product_margin_summary(
    data_frame: pd.DataFrame,
    product_column: str | None,
    sales_column: str | None,
    profit_column: str | None,
) -> pd.DataFrame:
    if (
        data_frame is None
        or data_frame.empty
        or not product_column
        or not sales_column
        or not profit_column
        or product_column not in data_frame.columns
        or sales_column not in data_frame.columns
        or profit_column not in data_frame.columns
    ):
        return pd.DataFrame()

    product_summary = (
        data_frame.groupby(product_column, dropna=False)
        .agg(
            sales=(sales_column, "sum"),
            profit=(profit_column, "sum"),
        )
        .reset_index()
    )

    product_summary["gross_margin_pct"] = np.where(
        product_summary["sales"] != 0,
        product_summary["profit"] / product_summary["sales"] * 100,
        0,
    )

    return product_summary


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data(show_spinner=False)
def load_processed_files() -> dict[str, pd.DataFrame]:
    loaded_data = {}

    for key, file_name in DATA_FILES.items():
        file_path = DATA_DIR / file_name
        loaded_data[key] = read_csv_safely(file_path)

    return loaded_data


data = load_processed_files()

main_df = data.get("final_feature", pd.DataFrame()).copy()

if main_df.empty:
    main_df = data.get("feature_engineered", pd.DataFrame()).copy()

if main_df.empty:
    main_df = data.get("cleaned", pd.DataFrame()).copy()

if main_df.empty:
    st.error(
        "No main dataset found. Please check these files inside data/processed: "
        "final_feature_engineered_dataset.csv, feature_engineered_nassau_candy.csv, cleaned_nassau_candy.csv"
    )
    st.stop()

# =========================================================
# COLUMN DETECTION
# =========================================================
date_col = find_col(
    main_df,
    ["Order Date", "Order_Date", "Date", "OrderDate"],
)

division_col = find_col(
    main_df,
    ["Division", "Category", "Product Division", "Product_Division"],
)

product_col = find_col(
    main_df,
    ["Product Name", "Product_Name", "Product", "Item", "ProductName"],
)

sales_col = find_col(
    main_df,
    ["Sales", "Revenue", "Total Sales", "Total_Sales", "Product Sales"],
)

cost_col = find_col(
    main_df,
    ["Cost", "Total Cost", "Total_Cost", "Product Cost"],
)

profit_col = find_col(
    main_df,
    [
        "Gross Profit",
        "Gross_Profit",
        "Profit",
        "Total Profit",
        "Total_Profit",
        "Product Profit",
    ],
)

units_col = find_col(
    main_df,
    ["Units", "Quantity", "Qty", "Order Quantity", "Order_Quantity", "Units Sold"],
)

# Date conversion
if date_col and date_col in main_df.columns:
    main_df[date_col] = pd.to_datetime(main_df[date_col], errors="coerce")

# Numeric conversion
for numeric_col in [sales_col, cost_col, profit_col, units_col]:
    if numeric_col and numeric_col in main_df.columns:
        main_df[numeric_col] = to_num(main_df[numeric_col]).fillna(0)

# Create profit if missing
if not profit_col and sales_col and cost_col:
    main_df["Calculated Profit"] = main_df[sales_col] - main_df[cost_col]
    profit_col = "Calculated Profit"
    
# =========================================================
# CENTRAL FILTER STATE SYSTEM
# Initialize single source of truth for desktop + mobile filters
# =========================================================

def p2_get_date_bounds() -> tuple:
    """Return safe min/max date bounds for Project 2 filters."""
    if date_col and date_col in main_df.columns and main_df[date_col].notna().any():
        return main_df[date_col].min().date(), main_df[date_col].max().date()

    today_value = pd.Timestamp.today().date()
    return today_value, today_value


def p2_clamp_date_range(start_value, end_value):
    """Keep selected dates inside the available dataset date range."""
    min_date_value, max_date_value = p2_get_date_bounds()

    if start_value is None or end_value is None:
        return min_date_value, max_date_value

    if start_value < min_date_value:
        start_value = min_date_value

    if end_value > max_date_value:
        end_value = max_date_value

    if start_value > end_value:
        return min_date_value, max_date_value

    return start_value, end_value


def p2_get_date_filtered_df(start_value, end_value) -> pd.DataFrame:
    """Return dataframe filtered by selected date range only."""
    working_df = main_df.copy()

    if date_col and date_col in working_df.columns and working_df[date_col].notna().any():
        start_value, end_value = p2_clamp_date_range(start_value, end_value)

        working_df = working_df[
            (working_df[date_col].dt.date >= start_value)
            & (working_df[date_col].dt.date <= end_value)
        ]

    return working_df


def p2_get_division_options(start_value, end_value) -> list[str]:
    """Return division options based on selected date range."""
    date_scoped_df = p2_get_date_filtered_df(start_value, end_value)

    if division_col and division_col in date_scoped_df.columns and not date_scoped_df.empty:
        division_values = (
            date_scoped_df[division_col]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        return ["All Divisions"] + division_values

    return ["All Divisions"]


def p2_get_division_filtered_df(start_value, end_value, division_value: str) -> pd.DataFrame:
    """Return dataframe filtered by date range and selected division."""
    working_df = p2_get_date_filtered_df(start_value, end_value)

    if (
        division_col
        and division_col in working_df.columns
        and division_value != "All Divisions"
    ):
        working_df = working_df[
            working_df[division_col].astype(str) == str(division_value)
        ]

    return working_df


def p2_get_product_options(start_value, end_value, division_value: str) -> list[str]:
    """Return product options based on selected date range + division."""
    division_scoped_df = p2_get_division_filtered_df(
        start_value,
        end_value,
        division_value,
    )

    if product_col and product_col in division_scoped_df.columns and not division_scoped_df.empty:
        product_values = (
            division_scoped_df[product_col]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        return ["All Products"] + product_values

    return ["All Products"]


# -----------------------------
# Initialize final filter state
# -----------------------------
p2_min_date, p2_max_date = p2_get_date_bounds()

p2_default_filter_state = {
    "p2_active_filter_source": "desktop",
    "p2_final_start_date": p2_min_date,
    "p2_final_end_date": p2_max_date,
    "p2_final_division": "All Divisions",
    "p2_final_product": "All Products",
    "p2_final_margin_threshold": 30,
}

for state_key, default_value in p2_default_filter_state.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# -----------------------------
# Validate existing final state after reruns
# -----------------------------

# Date safety
p2_valid_start_date, p2_valid_end_date = p2_clamp_date_range(
    st.session_state.get("p2_final_start_date", p2_min_date),
    st.session_state.get("p2_final_end_date", p2_max_date),
)

st.session_state["p2_final_start_date"] = p2_valid_start_date
st.session_state["p2_final_end_date"] = p2_valid_end_date

# Division safety
p2_valid_division_options = p2_get_division_options(
    st.session_state["p2_final_start_date"],
    st.session_state["p2_final_end_date"],
)

if st.session_state.get("p2_final_division") not in p2_valid_division_options:
    st.session_state["p2_final_division"] = "All Divisions"

# Product safety
p2_valid_product_options = p2_get_product_options(
    st.session_state["p2_final_start_date"],
    st.session_state["p2_final_end_date"],
    st.session_state["p2_final_division"],
)

if st.session_state.get("p2_final_product") not in p2_valid_product_options:
    st.session_state["p2_final_product"] = "All Products"

# Margin threshold safety
try:
    p2_threshold_value = int(st.session_state.get("p2_final_margin_threshold", 30))
except Exception:
    p2_threshold_value = 30

p2_threshold_value = max(0, min(100, p2_threshold_value))
st.session_state["p2_final_margin_threshold"] = p2_threshold_value

# =========================================================
# SIDEBAR FILTERS — DESKTOP SOURCE
# Updates central filter state from desktop sidebar
# =========================================================
with st.sidebar:
    # =========================================================
    # SIDEBAR TOP BRAND AREA
    # Clean structure: logo without card + compact project context
    # =========================================================
    with st.container(key="sidebar_top_brand"):
        if UNIFIED_LOGO.exists():
            st.image(str(UNIFIED_LOGO), width=220)
        else:
            st.markdown("### Unified Mentor")

    with st.container(key="sidebar_project_summary"):
        st.markdown("**Nassau Candy Profitability Margin Performance Analysis Dashboard**")
        st.caption(
            "Product margin, cost, and profit concentration insights."
        )

    st.divider()

    with st.container(key="sidebar_filter_header"):
        st.markdown("### Filters")
        st.caption("Adjust the analysis scope. Every KPI, chart, and table updates dynamically.")

    with st.container(key="sidebar_filter_controls"):

        # -----------------------------
        # 1. Desktop Date Range
        # -----------------------------
        if date_col and main_df[date_col].notna().any():
            desktop_default_dates = (
                st.session_state["p2_final_start_date"],
                st.session_state["p2_final_end_date"],
            )

            desktop_selected_dates = st.date_input(
                "Date Range",
                value=desktop_default_dates,
                min_value=p2_min_date,
                max_value=p2_max_date,
                key="p2_desktop_date_range_filter",
            )

            if (
                isinstance(desktop_selected_dates, tuple)
                and len(desktop_selected_dates) == 2
            ):
                desktop_start_date, desktop_end_date = desktop_selected_dates
            else:
                desktop_start_date, desktop_end_date = p2_min_date, p2_max_date
        else:
            desktop_start_date, desktop_end_date = p2_min_date, p2_max_date
            st.info("Date column not detected.")

        desktop_start_date, desktop_end_date = p2_clamp_date_range(
            desktop_start_date,
            desktop_end_date,
        )

        # -----------------------------
        # 2. Desktop Division Filter
        # Division options depend on selected desktop date range
        # -----------------------------
        desktop_division_options = p2_get_division_options(
            desktop_start_date,
            desktop_end_date,
        )

        current_desktop_division = st.session_state.get(
            "p2_final_division",
            "All Divisions",
        )

        if current_desktop_division not in desktop_division_options:
            current_desktop_division = "All Divisions"

        desktop_division_index = (
            desktop_division_options.index(current_desktop_division)
            if current_desktop_division in desktop_division_options
            else 0
        )

        desktop_selected_division = st.selectbox(
            "Division",
            options=desktop_division_options,
            index=desktop_division_index,
            key="p2_desktop_division_filter",
        )

        # -----------------------------
        # 3. Desktop Margin Threshold
        # Risk benchmark, not direct data filter
        # -----------------------------
        desktop_margin_threshold = st.slider(
            "Margin Threshold (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state["p2_final_margin_threshold"]),
            step=1,
            key="p2_desktop_margin_threshold_filter",
        )

        # -----------------------------
        # 4. Desktop Product Search
        # Product options depend on selected Date Range + Division
        # -----------------------------
        desktop_product_options = p2_get_product_options(
            desktop_start_date,
            desktop_end_date,
            desktop_selected_division,
        )

        current_desktop_product = st.session_state.get(
            "p2_final_product",
            "All Products",
        )

        if current_desktop_product not in desktop_product_options:
            current_desktop_product = "All Products"

        desktop_product_index = (
            desktop_product_options.index(current_desktop_product)
            if current_desktop_product in desktop_product_options
            else 0
        )

        desktop_selected_product = st.selectbox(
            "Product Search",
            options=desktop_product_options,
            index=desktop_product_index,
            key="p2_desktop_product_filter",
        )

        # -----------------------------
        # Update central state from desktop filters ONLY when desktop changes
        # Prevents hidden mobile panel from fighting with desktop state.
        # -----------------------------
        desktop_filter_payload = {
            "start_date": desktop_start_date,
            "end_date": desktop_end_date,
            "division": desktop_selected_division,
            "product": desktop_selected_product,
            "margin_threshold": int(desktop_margin_threshold),
        }

        desktop_previous_payload = st.session_state.get("_p2_desktop_filter_payload")

        if desktop_previous_payload is None or desktop_filter_payload != desktop_previous_payload:
            st.session_state["p2_active_filter_source"] = "desktop"
            st.session_state["p2_final_start_date"] = desktop_filter_payload["start_date"]
            st.session_state["p2_final_end_date"] = desktop_filter_payload["end_date"]
            st.session_state["p2_final_division"] = desktop_filter_payload["division"]
            st.session_state["p2_final_product"] = desktop_filter_payload["product"]
            st.session_state["p2_final_margin_threshold"] = desktop_filter_payload["margin_threshold"]

        st.session_state["_p2_desktop_filter_payload"] = desktop_filter_payload

    with st.container(key="sidebar_footer_note"):
        st.caption("Portfolio dashboard • Streamlit + Plotly")
        st.caption("Nassau Candy profitability analysis")
        
# =========================================================
# MOBILE FILTER PANEL — MOBILE SOURCE
# Updates Project 2 central filter state only when mobile filters change
# =========================================================
with st.container(key="mobile_filter_panel"):
    with st.expander("📱 Dashboard Filters", expanded=False):
        st.caption(
            "Adjust the analysis scope to explore product margins, cost efficiency, "
            "contribution mix, and concentration risk."
        )

        # -----------------------------
        # 1. Mobile Date Range
        # -----------------------------
        mobile_default_dates = (
            st.session_state["p2_final_start_date"],
            st.session_state["p2_final_end_date"],
        )

        mobile_selected_dates = st.date_input(
            "Date Range",
            value=mobile_default_dates,
            min_value=p2_min_date,
            max_value=p2_max_date,
            key="p2_mobile_date_range_filter",
        )

        if (
            isinstance(mobile_selected_dates, tuple)
            and len(mobile_selected_dates) == 2
        ):
            mobile_start_date, mobile_end_date = mobile_selected_dates
        else:
            mobile_start_date, mobile_end_date = p2_min_date, p2_max_date

        mobile_start_date, mobile_end_date = p2_clamp_date_range(
            mobile_start_date,
            mobile_end_date,
        )

        # -----------------------------
        # 2. Mobile Division Filter
        # Division options depend on selected mobile date range
        # -----------------------------
        mobile_division_options = p2_get_division_options(
            mobile_start_date,
            mobile_end_date,
        )

        current_mobile_division = st.session_state.get(
            "p2_final_division",
            "All Divisions",
        )

        if current_mobile_division not in mobile_division_options:
            current_mobile_division = "All Divisions"

        mobile_division_index = (
            mobile_division_options.index(current_mobile_division)
            if current_mobile_division in mobile_division_options
            else 0
        )

        mobile_selected_division = st.selectbox(
            "Division",
            options=mobile_division_options,
            index=mobile_division_index,
            key="p2_mobile_division_filter",
        )

        # -----------------------------
        # 3. Mobile Margin Threshold
        # Risk benchmark, not direct data filter
        # -----------------------------
        mobile_margin_threshold = st.slider(
            "Margin Threshold (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state["p2_final_margin_threshold"]),
            step=1,
            key="p2_mobile_margin_threshold_filter",
        )

        # -----------------------------
        # 4. Mobile Product Search
        # Product options depend on selected mobile Date Range + Division
        # -----------------------------
        mobile_product_options = p2_get_product_options(
            mobile_start_date,
            mobile_end_date,
            mobile_selected_division,
        )

        current_mobile_product = st.session_state.get(
            "p2_final_product",
            "All Products",
        )

        if current_mobile_product not in mobile_product_options:
            current_mobile_product = "All Products"

        mobile_product_index = (
            mobile_product_options.index(current_mobile_product)
            if current_mobile_product in mobile_product_options
            else 0
        )

        mobile_selected_product = st.selectbox(
            "Product Search",
            options=mobile_product_options,
            index=mobile_product_index,
            key="p2_mobile_product_filter",
        )

        # -----------------------------
        # Update central state from mobile filters ONLY when mobile changes
        # First render stores payload only; it does not overwrite desktop state.
        # -----------------------------
        mobile_filter_payload = {
            "start_date": mobile_start_date,
            "end_date": mobile_end_date,
            "division": mobile_selected_division,
            "product": mobile_selected_product,
            "margin_threshold": int(mobile_margin_threshold),
        }

        mobile_previous_payload = st.session_state.get("_p2_mobile_filter_payload")

        if mobile_previous_payload is None:
            st.session_state["_p2_mobile_filter_payload"] = mobile_filter_payload

        elif mobile_filter_payload != mobile_previous_payload:
            st.session_state["p2_active_filter_source"] = "mobile"
            st.session_state["p2_final_start_date"] = mobile_filter_payload["start_date"]
            st.session_state["p2_final_end_date"] = mobile_filter_payload["end_date"]
            st.session_state["p2_final_division"] = mobile_filter_payload["division"]
            st.session_state["p2_final_product"] = mobile_filter_payload["product"]
            st.session_state["p2_final_margin_threshold"] = mobile_filter_payload["margin_threshold"]

            st.session_state["_p2_mobile_filter_payload"] = mobile_filter_payload

        else:
            st.session_state["_p2_mobile_filter_payload"] = mobile_filter_payload


# =========================================================
# FINAL FILTER VARIABLES
# Single source of truth for KPI, charts, tables, tabs, and insights
# =========================================================
start_date = st.session_state["p2_final_start_date"]
end_date = st.session_state["p2_final_end_date"]
selected_division = st.session_state["p2_final_division"]
selected_product = st.session_state["p2_final_product"]
margin_threshold = st.session_state["p2_final_margin_threshold"]


# =========================================================
# STEP 5 — PROJECT 2 FINAL FILTER STATE STABILITY GUARD
# Prevents invalid filter state after desktop/mobile switching
# =========================================================

# Date safety
start_date, end_date = p2_clamp_date_range(start_date, end_date)
st.session_state["p2_final_start_date"] = start_date
st.session_state["p2_final_end_date"] = end_date

# Division safety based on selected date range
valid_division_options = p2_get_division_options(
    start_date,
    end_date,
)

if selected_division not in valid_division_options:
    selected_division = "All Divisions"
    st.session_state["p2_final_division"] = selected_division

# Product safety based on selected date range + division
valid_product_options = p2_get_product_options(
    start_date,
    end_date,
    selected_division,
)

if selected_product not in valid_product_options:
    selected_product = "All Products"
    st.session_state["p2_final_product"] = selected_product

# Margin threshold safety
try:
    margin_threshold = int(margin_threshold)
except Exception:
    margin_threshold = 30

margin_threshold = max(0, min(100, margin_threshold))
st.session_state["p2_final_margin_threshold"] = margin_threshold

# Final scoped dataframes for existing FILTER DATA section
date_filtered_df = p2_get_date_filtered_df(
    start_date,
    end_date,
)

division_filtered_df = p2_get_division_filtered_df(
    start_date,
    end_date,
    selected_division,
)

# =========================================================
# FILTER DATA
# =========================================================
filtered_df = division_filtered_df.copy()

# Final product filter
if product_col and selected_product != "All Products":
    filtered_df = filtered_df[
        filtered_df[product_col].astype(str) == selected_product
    ]

if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust filters.")
    st.stop()

# =========================================================
# DYNAMIC KPI CALCULATIONS
# =========================================================
total_sales_all = main_df[sales_col].sum() if sales_col else 0
total_profit_all = main_df[profit_col].sum() if profit_col else 0

filtered_sales = filtered_df[sales_col].sum() if sales_col else 0
filtered_profit = filtered_df[profit_col].sum() if profit_col else 0
filtered_units = filtered_df[units_col].sum() if units_col else 0

gross_margin_pct = (filtered_profit / filtered_sales * 100) if filtered_sales else 0
profit_per_unit = (filtered_profit / filtered_units) if filtered_units else 0
revenue_contribution = (filtered_sales / total_sales_all * 100) if total_sales_all else 0
profit_contribution = (filtered_profit / total_profit_all * 100) if total_profit_all else 0

if date_col and sales_col and profit_col and filtered_df[date_col].notna().any():
    margin_time = (
        filtered_df.groupby(filtered_df[date_col].dt.date)
        .agg(
            sales=(sales_col, "sum"),
            profit=(profit_col, "sum"),
        )
        .reset_index()
    )

    margin_time["margin_pct"] = np.where(
        margin_time["sales"] != 0,
        margin_time["profit"] / margin_time["sales"] * 100,
        0,
    )

    margin_volatility = margin_time["margin_pct"].std()

    if pd.isna(margin_volatility):
        margin_volatility = 0
else:
    margin_volatility = 0

# =========================================================
# MARGIN THRESHOLD RISK LOGIC
# =========================================================
product_margin_summary = build_product_margin_summary(
    filtered_df,
    product_col,
    sales_col,
    profit_col,
)

if not product_margin_summary.empty:
    risk_products_df = product_margin_summary[
        product_margin_summary["gross_margin_pct"] < margin_threshold
    ].copy()

    risk_product_count = len(risk_products_df)
    revenue_at_risk = risk_products_df["sales"].sum()
    profit_at_risk = risk_products_df["profit"].sum()

    revenue_at_risk_pct = (revenue_at_risk / filtered_sales * 100) if filtered_sales else 0
    profit_at_risk_pct = (profit_at_risk / filtered_profit * 100) if filtered_profit else 0
else:
    risk_product_count = 0
    revenue_at_risk = 0
    profit_at_risk = 0
    revenue_at_risk_pct = 0
    profit_at_risk_pct = 0
    
# =========================================================
# STEP 1D — PRODUCT PROFITABILITY OVERVIEW DATA
# =========================================================
def build_product_profitability_summary(
    data_frame: pd.DataFrame,
    product_column: str | None,
    sales_column: str | None,
    profit_column: str | None,
    units_column: str | None,
    threshold_value: float,
) -> pd.DataFrame:
    if (
        data_frame is None
        or data_frame.empty
        or not product_column
        or not sales_column
        or not profit_column
        or product_column not in data_frame.columns
        or sales_column not in data_frame.columns
        or profit_column not in data_frame.columns
    ):
        return pd.DataFrame()

    agg_dict = {
        "sales": (sales_column, "sum"),
        "profit": (profit_column, "sum"),
    }

    if units_column and units_column in data_frame.columns:
        agg_dict["units"] = (units_column, "sum")

    product_summary = (
        data_frame.groupby(product_column, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={product_column: "product_name"})
    )

    if "units" not in product_summary.columns:
        product_summary["units"] = 0

    product_summary["gross_margin_pct"] = np.where(
        product_summary["sales"] != 0,
        product_summary["profit"] / product_summary["sales"] * 100,
        0,
    )

    product_summary["profit_per_unit"] = np.where(
        product_summary["units"] != 0,
        product_summary["profit"] / product_summary["units"],
        0,
    )

    product_summary["revenue_contribution_pct"] = np.where(
        filtered_sales != 0,
        product_summary["sales"] / filtered_sales * 100,
        0,
    )

    product_summary["profit_contribution_pct"] = np.where(
        filtered_profit != 0,
        product_summary["profit"] / filtered_profit * 100,
        0,
    )

    product_summary["margin_status"] = np.where(
        product_summary["gross_margin_pct"] < threshold_value,
        "Below Threshold",
        "Healthy Margin",
    )

    return product_summary.sort_values("profit", ascending=False)


product_profitability_df = build_product_profitability_summary(
    filtered_df,
    product_col,
    sales_col,
    profit_col,
    units_col,
    margin_threshold,
)

# Portfolio comparison data keeps single-product visuals meaningful.
portfolio_base_df = division_filtered_df.copy()

portfolio_profitability_df = build_product_profitability_summary(
    portfolio_base_df,
    product_col,
    sales_col,
    profit_col,
    units_col,
    margin_threshold,
)

# =========================================================
# STEP 2A — DIVISION PERFORMANCE DATA HELPERS
# =========================================================
def build_division_performance_summary(
    data_frame: pd.DataFrame,
    division_column: str | None,
    sales_column: str | None,
    profit_column: str | None,
) -> pd.DataFrame:
    if (
        data_frame is None
        or data_frame.empty
        or not division_column
        or not sales_column
        or not profit_column
        or division_column not in data_frame.columns
        or sales_column not in data_frame.columns
        or profit_column not in data_frame.columns
    ):
        return pd.DataFrame()

    division_summary = (
        data_frame.groupby(division_column, dropna=False)
        .agg(
            sales=(sales_column, "sum"),
            profit=(profit_column, "sum"),
        )
        .reset_index()
        .rename(columns={division_column: "division_name"})
    )

    division_summary["gross_margin_pct"] = np.where(
        division_summary["sales"] != 0,
        division_summary["profit"] / division_summary["sales"] * 100,
        0,
    )

    division_summary["revenue_contribution_pct"] = np.where(
        division_summary["sales"].sum() != 0,
        division_summary["sales"] / division_summary["sales"].sum() * 100,
        0,
    )

    division_summary["profit_contribution_pct"] = np.where(
        division_summary["profit"].sum() != 0,
        division_summary["profit"] / division_summary["profit"].sum() * 100,
        0,
    )

    return division_summary.sort_values("profit", ascending=False)

# =========================================================
# STEP 2A — DIVISION PERFORMANCE DATA
# =========================================================
division_performance_df = build_division_performance_summary(
    filtered_df,
    division_col,
    sales_col,
    profit_col,
)

division_distribution_df = filtered_df.copy()

if (
    division_col
    and sales_col
    and profit_col
    and division_col in division_distribution_df.columns
    and sales_col in division_distribution_df.columns
    and profit_col in division_distribution_df.columns
):
    division_distribution_df["row_margin_pct"] = np.where(
        division_distribution_df[sales_col] != 0,
        division_distribution_df[profit_col] / division_distribution_df[sales_col] * 100,
        0,
    )
else:
    division_distribution_df = pd.DataFrame()


# =========================================================
# STEP 3A — COST VS MARGIN DIAGNOSTICS DATA HELPERS
# =========================================================
def build_cost_diagnostics_summary(
    data_frame: pd.DataFrame,
    product_column: str | None,
    sales_column: str | None,
    cost_column: str | None,
    profit_column: str | None,
    units_column: str | None,
    threshold_value: float,
) -> pd.DataFrame:
    if (
        data_frame is None
        or data_frame.empty
        or not product_column
        or not sales_column
        or not profit_column
        or product_column not in data_frame.columns
        or sales_column not in data_frame.columns
        or profit_column not in data_frame.columns
    ):
        return pd.DataFrame()

    working_df = data_frame.copy()

    if cost_column and cost_column in working_df.columns:
        working_df["_diagnostic_cost"] = working_df[cost_column]
    else:
        working_df["_diagnostic_cost"] = working_df[sales_column] - working_df[profit_column]

    agg_dict = {
        "sales": (sales_column, "sum"),
        "cost": ("_diagnostic_cost", "sum"),
        "profit": (profit_column, "sum"),
        "records": (product_column, "count"),
    }

    if units_column and units_column in working_df.columns:
        agg_dict["units"] = (units_column, "sum")

    cost_summary = (
        working_df.groupby(product_column, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={product_column: "product_name"})
    )

    if "units" not in cost_summary.columns:
        cost_summary["units"] = 0

    cost_summary["gross_margin_pct"] = np.where(
        cost_summary["sales"] != 0,
        cost_summary["profit"] / cost_summary["sales"] * 100,
        0,
    )

    cost_summary["cost_to_sales_pct"] = np.where(
        cost_summary["sales"] != 0,
        cost_summary["cost"] / cost_summary["sales"] * 100,
        0,
    )

    cost_summary["profit_per_unit"] = np.where(
        cost_summary["units"] != 0,
        cost_summary["profit"] / cost_summary["units"],
        0,
    )

    cost_summary["margin_status"] = np.where(
        cost_summary["gross_margin_pct"] < threshold_value,
        "Below Threshold",
        "Healthy Margin",
    )

    cost_summary["risk_flag"] = np.select(
        [
            cost_summary["gross_margin_pct"] < threshold_value,
            cost_summary["cost_to_sales_pct"] >= 75,
            cost_summary["profit"] <= 0,
        ],
        [
            "Margin Risk",
            "High Cost Load",
            "Profit Risk",
        ],
        default="Stable",
    )

    cost_summary["diagnostic_priority"] = np.select(
        [
            cost_summary["risk_flag"].eq("Margin Risk"),
            cost_summary["risk_flag"].eq("High Cost Load"),
            cost_summary["risk_flag"].eq("Profit Risk"),
        ],
        [1, 2, 3],
        default=4,
    )

    return cost_summary.sort_values(
        ["diagnostic_priority", "gross_margin_pct", "cost_to_sales_pct"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


cost_diagnostics_df = build_cost_diagnostics_summary(
    filtered_df,
    product_col,
    sales_col,
    cost_col,
    profit_col,
    units_col,
    margin_threshold,
)


# =========================================================
# STEP 4A — PROFIT CONCENTRATION ANALYSIS DATA HELPERS
# =========================================================
def build_profit_concentration_summary(
    data_frame: pd.DataFrame,
    product_column: str | None,
    sales_column: str | None,
    profit_column: str | None,
    threshold_value: float,
) -> pd.DataFrame:
    if (
        data_frame is None
        or data_frame.empty
        or not product_column
        or not sales_column
        or not profit_column
        or product_column not in data_frame.columns
        or sales_column not in data_frame.columns
        or profit_column not in data_frame.columns
    ):
        return pd.DataFrame()

    pareto_df = (
        data_frame.groupby(product_column, dropna=False)
        .agg(
            sales=(sales_column, "sum"),
            profit=(profit_column, "sum"),
            records=(product_column, "count"),
        )
        .reset_index()
        .rename(columns={product_column: "product_name"})
    )

    pareto_df = pareto_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    pareto_df = pareto_df[pareto_df["profit"] > 0].copy()

    if pareto_df.empty:
        return pd.DataFrame()

    total_profit_value = pareto_df["profit"].sum()
    total_sales_value = pareto_df["sales"].sum()

    pareto_df = pareto_df.sort_values("profit", ascending=False).reset_index(drop=True)
    pareto_df["rank"] = np.arange(1, len(pareto_df) + 1)
    pareto_df["profit_contribution_pct"] = np.where(
        total_profit_value != 0,
        pareto_df["profit"] / total_profit_value * 100,
        0,
    )
    pareto_df["sales_contribution_pct"] = np.where(
        total_sales_value != 0,
        pareto_df["sales"] / total_sales_value * 100,
        0,
    )
    pareto_df["cumulative_profit"] = pareto_df["profit"].cumsum()
    pareto_df["cumulative_profit_pct"] = np.where(
        total_profit_value != 0,
        pareto_df["cumulative_profit"] / total_profit_value * 100,
        0,
    )
    pareto_df["gross_margin_pct"] = np.where(
        pareto_df["sales"] != 0,
        pareto_df["profit"] / pareto_df["sales"] * 100,
        0,
    )
    pareto_df["margin_status"] = np.where(
        pareto_df["gross_margin_pct"] < threshold_value,
        "Below Threshold",
        "Healthy Margin",
    )
    pareto_df["pareto_tier"] = np.where(
        pareto_df["cumulative_profit_pct"] <= 80,
        "Core 80% Driver",
        "Long-Tail Profit",
    )

    if not (pareto_df["pareto_tier"] == "Core 80% Driver").any():
        pareto_df.loc[pareto_df.index[0], "pareto_tier"] = "Core 80% Driver"

    pareto_df["dependency_status"] = np.select(
        [
            pareto_df["profit_contribution_pct"] >= 35,
            pareto_df["profit_contribution_pct"] >= 20,
            pareto_df["pareto_tier"].eq("Core 80% Driver"),
        ],
        [
            "High Dependency",
            "Moderate Dependency",
            "Core Driver",
        ],
        default="Distributed",
    )

    return pareto_df


profit_concentration_df = build_profit_concentration_summary(
    filtered_df,
    product_col,
    sales_col,
    profit_col,
    margin_threshold,
)

# =========================================================
# HEADER / HERO SECTION
# Desktop and mobile separated for stable responsive layout
# =========================================================

# Desktop Header
with st.container(key="desktop_header_card"):
    header_logo_col, header_title_col = st.columns([0.24, 0.76], vertical_alignment="center")

    with header_logo_col:
        if NASSAU_LOGO.exists():
            trimmed_nassau_logo = get_trimmed_logo_path(str(NASSAU_LOGO))
            st.image(trimmed_nassau_logo, width=230)
        else:
            st.markdown("### Nassau Candy")

    with header_title_col:
        st.markdown("## Product Line Profitability & Margin Performance Analysis")


# Mobile Header
with st.container(key="mobile_header_card"):
    if NASSAU_LOGO.exists():
        trimmed_nassau_logo = get_trimmed_logo_path(str(NASSAU_LOGO))
        st.image(trimmed_nassau_logo, width=235)
    else:
        st.markdown("### Nassau Candy")

    st.markdown("## Product Line Profitability & Margin Performance Analysis")

# =========================================================
# STEP 5A — FINAL EXECUTIVE INTRO
# =========================================================
st.success(
    "Analyze Nassau Candy’s product-line profitability through a dynamic view of margins, cost efficiency, product contribution, and profit concentration. This dashboard helps identify high-performing products, margin-risk items, division-level profitability patterns, and dependency risks across the portfolio. Use the filters to explore product, division, and threshold-based performance for stronger executive decision-making."
)

# =========================================================
# STEP 1B — DYNAMIC KPI ROW ONLY
# STEP 7E: FINAL KPI + RISK MONITOR MICRO-POLISH
# Safe mode: presentation containers and text hierarchy only. Calculations unchanged.
# =========================================================
with st.container(key="kpi_executive_zone"):
    st.subheader("Key Performance Indicators")
    st.caption(
        "Executive margin, unit profit, contribution, and volatility signals based on the current filter context."
    )

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)

    with kpi_1:
        st.metric(
            label="Gross Margin (%)",
            value=fmt_pct(gross_margin_pct, 1),
            delta="Gross Profit ÷ Sales",
        )

    with kpi_2:
        st.metric(
            label="Profit per Unit",
            value=fmt_money(profit_per_unit, 2),
            delta="Gross Profit ÷ Units",
        )

    with kpi_3:
        st.metric(
            label="Revenue Contribution",
            value=fmt_pct(revenue_contribution, 1),
            delta="Filtered Sales ÷ Total Sales",
        )

    with kpi_4:
        st.metric(
            label="Profit Contribution",
            value=fmt_pct(profit_contribution, 1),
            delta="Filtered Profit ÷ Total Profit",
        )

    with kpi_5:
        st.metric(
            label="Margin Volatility",
            value=f"{fmt_num(margin_volatility, 2)} pts",
            delta="Margin variability over time",
        )

# =========================================================
# STEP 1C — MARGIN THRESHOLD RISK MONITOR
# STEP 7E: FINAL KPI + RISK MONITOR MICRO-POLISH
# Safe mode: presentation containers and text hierarchy only. Calculations unchanged.
# =========================================================
with st.container(key="risk_monitor_zone"):
    st.markdown("---")
    st.subheader("Margin Threshold Risk Monitor")
    st.caption(
        "Risk exposure updates dynamically by selected margin benchmark, date range, division, and product filters."
    )

    risk_1, risk_2, risk_3, risk_4 = st.columns(4)

    with risk_1:
        st.metric(
            label="Products Below Threshold",
            value=f"{risk_product_count:,}",
            delta=f"Below {margin_threshold}%",
        )

    with risk_2:
        st.metric(
            label="Revenue at Risk",
            value=fmt_money(revenue_at_risk, 0),
            delta=f"{fmt_pct(revenue_at_risk_pct, 1)} of filtered revenue",
        )

    with risk_3:
        st.metric(
            label="Profit at Risk",
            value=fmt_money(profit_at_risk, 0),
            delta=f"{fmt_pct(profit_at_risk_pct, 1)} of filtered profit",
        )

    with risk_4:
        st.metric(
            label="Selected Threshold",
            value=fmt_pct(margin_threshold, 0),
            delta="Risk flag benchmark",
        )
    
# =========================================================
# STEP 1E — DASHBOARD MODULES TAB SHELL
# =========================================================
st.markdown("---")
st.subheader("Dashboard Modules")
st.caption(
    "Explore each required dashboard module separately. All modules will stay connected to the selected filters."
)

tab_product, tab_division, tab_cost, tab_pareto = st.tabs(
    [
        "Product Profitability",
        "Division Performance",
        "Cost Diagnostics",
        "Profit Concentration",
    ]
)

# =========================================================
# TAB 1 — PRODUCT PROFITABILITY OVERVIEW
# =========================================================
with tab_product:
    st.subheader("Product Profitability Overview")
    st.caption(
        "Dynamic profit contribution visuals, selected-product context, and product-level margin leaderboard based on the current filters."
    )

    if product_profitability_df.empty:
        st.warning("Product profitability data is not available for the selected filters.")
    else:
        _, portfolio_product_color_map = apply_dynamic_product_colors(
            portfolio_profitability_df.copy()
        )
        product_profitability_df, product_color_map = apply_dynamic_product_colors(
            product_profitability_df
        )
        product_color_map = {**portfolio_product_color_map, **product_color_map}

        single_product_view = is_single_product_mode(
            selected_product,
            product_profitability_df,
        )

        chart_height = get_product_chart_height(single_product_view)

        if single_product_view:
            insight_text = build_single_product_insight(
                product_profitability_df,
                margin_threshold,
            )

            if product_profitability_df.iloc[0]["margin_status"] == "Below Threshold":
                st.warning(insight_text)
            else:
                st.success(insight_text)

        top_profit_df = product_profitability_df.head(10).copy()
        top_profit_df["product_short"] = top_profit_df["product_name"].apply(
            lambda x: compact_product_label(x, 28)
        )

        margin_leaderboard_df = (
            product_profitability_df
            .sort_values("gross_margin_pct", ascending=False)
            .head(10)
            .copy()
        )

        margin_leaderboard_df["product_short"] = margin_leaderboard_df["product_name"].apply(
            lambda x: compact_product_label(x, 28)
        )

        overview_1, overview_2, overview_3 = st.columns(3)

        with overview_1:
            st.metric(
                label="Products in View",
                value=f"{len(product_profitability_df):,}",
                delta="Filtered product count",
            )

        with overview_2:
            best_product = (
                product_profitability_df.iloc[0]["product_name"]
                if not product_profitability_df.empty
                else "N/A"
            )

            st.metric(
                label="Top Profit Product",
                value=short_label(best_product, 24),
                delta="Highest filtered profit",
            )

        with overview_3:
            avg_product_margin = product_profitability_df["gross_margin_pct"].mean()

            st.metric(
                label="Avg Product Margin",
                value=fmt_pct(avg_product_margin, 1),
                delta="Average across products",
            )

        chart_col_1, chart_col_2 = st.columns([1.2, 0.9])

        with chart_col_1:
            st.markdown("#### Profit Contribution by Product")
            
            st.caption(
                "Gross profit contribution by visible products, ranked from highest to lowest based on the selected filters. This view helps identify which products are driving the strongest profit impact, which items have weaker contribution, and where the portfolio may be depending too heavily on a small group of products. Use it to compare product-level profitability strength before reviewing margin status and contribution share."
            )

            profit_chart_df = top_profit_df.sort_values("profit", ascending=True).copy()
            profit_chart_df["profit_label"] = profit_chart_df["profit"].apply(fmt_money_compact)

            profit_chart = px.bar(
                profit_chart_df,
                x="profit",
                y="product_short",
                orientation="h",
                text="profit_label",
                color="product_name",
                custom_data=[
                    "product_name",
                    "sales",
                    "gross_margin_pct",
                    "profit_contribution_pct",
                    "margin_status",
                ],
                labels={
                    "profit": "Gross Profit",
                    "product_short": "Product",
                    "product_name": "Product",
                },
                color_discrete_map=product_color_map,
            )

            profit_chart.update_traces(
                texttemplate="%{text}",
                textposition="outside",
                cliponaxis=False,
                marker_line_color="rgba(255,255,255,0.82)",
                marker_line_width=1.1,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Gross Profit: $%{x:,.0f}<br>"
                    "Sales: $%{customdata[1]:,.0f}<br>"
                    "Gross Margin: %{customdata[2]:.1f}%<br>"
                    "Profit Contribution: %{customdata[3]:.1f}%<br>"
                    "Margin Status: %{customdata[4]}"
                    "<extra></extra>"
                ),
            )

            profit_chart.update_layout(
                height=chart_height,
                margin=dict(l=12, r=86, t=18, b=24),
                xaxis_title="Gross Profit",
                yaxis_title=None,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#24324a", size=12),
                bargap=0.36 if single_product_view else 0.24,
            )

            profit_chart = apply_nc_plotly_theme(
                profit_chart,
                height=chart_height + 28,
                margin=dict(l=24, r=148, t=24, b=60),
                showlegend=False,
            )
            profit_chart.update_xaxes(tickprefix="$")
            profit_chart.update_yaxes(
                tickfont=dict(size=10.5, color=NC_COLORS["axis"]),
                categoryorder="array",
                categoryarray=profit_chart_df["product_short"].tolist(),
            )

            st.plotly_chart(profit_chart, use_container_width=True, config=PLOTLY_CONFIG)

        with chart_col_2:
            st.markdown("#### Profit Contribution Share")
            st.caption(
                "Profit contribution mix across visible products, showing how total filtered profit is distributed across the portfolio. This view helps separate core profit drivers from long-tail products, making it easier to understand whether profitability is well-balanced or concentrated in only a few items."
            )
            st.caption(get_donut_context_caption(single_product_view))

            donut_df = product_profitability_df.sort_values(
                "profit_contribution_pct",
                ascending=False,
            ).copy()

            donut_color_map = product_color_map.copy()

            if selected_product != "All Products" and len(product_profitability_df) == 1:
                donut_df = build_selected_vs_remaining_profit_share(
                    selected_product,
                    product_profitability_df,
                    portfolio_profitability_df,
                )

                selected_color = product_status_color(
                    selected_product,
                    product_profitability_df.iloc[0].get("margin_status", "Healthy Margin"),
                )
                donut_df["product_short"] = donut_df["product_name"].apply(
                    lambda x: compact_product_label(x, 20)
                )
                donut_color_map = {
                    selected_product: selected_color,
                    compact_legend_label(selected_product, 20): selected_color,
                    "Remaining Portfolio": NC_COLORS["mint"],
                }

            elif len(donut_df) > 8:
                top_donut = donut_df.head(7).copy()
                top_donut["product_short"] = top_donut["product_name"].apply(
                    lambda x: compact_product_label(x, 22)
                )

                other_profit = donut_df.iloc[7:]["profit"].sum()
                other_sales = donut_df.iloc[7:]["sales"].sum()
                other_margin = (
                    other_profit / other_sales * 100
                    if other_sales
                    else 0
                )

                other_row = pd.DataFrame(
                    [
                        {
                            "product_name": "Other Products",
                            "product_short": "Other Products",
                            "sales": other_sales,
                            "profit": other_profit,
                            "gross_margin_pct": other_margin,
                            "profit_contribution_pct": (
                                other_profit / filtered_profit * 100
                                if filtered_profit
                                else 0
                            ),
                            "margin_status": "Portfolio Mix",
                        }
                    ]
                )

                donut_df = pd.concat([top_donut, other_row], ignore_index=True)
                donut_color_map["Other Products"] = "#79c7b8"
            else:
                donut_df["product_short"] = donut_df["product_name"].apply(
                    lambda x: compact_product_label(x, 22)
                )

            donut_df["legend_label"] = donut_df["product_name"].apply(
                lambda x: build_donut_legend_label(x, single_product_view)
            )

            donut_df["donut_context"] = np.where(
                single_product_view,
                "Selected product vs remaining filtered portfolio",
                "Filtered profit contribution mix",
            )

            donut_chart = px.pie(
                donut_df,
                names="legend_label",
                values="profit",
                hole=0.58,
                custom_data=[
                    "product_name",
                    "profit_contribution_pct",
                    "gross_margin_pct",
                    "sales",
                    "donut_context",
                ],
                color="product_name",
                color_discrete_map=donut_color_map,
            )

            donut_chart.update_traces(
                textposition="inside",
                textinfo="percent",
                texttemplate="%{percent}",
                insidetextorientation="horizontal",
                textfont=dict(size=11, color="#24324a"),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[4]}<br>"
                    "Gross Profit: <b>$%{value:,.0f}</b><br>"
                    "Profit Contribution: %{customdata[1]:.1f}%<br>"
                    "Gross Margin: %{customdata[2]:.1f}%<br>"
                    "Sales: $%{customdata[3]:,.0f}"
                    "<extra></extra>"
                ),
            )

            donut_chart.update_layout(
                height=chart_height,
                margin=dict(l=14, r=14, t=28, b=42),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.24 if single_product_view else -0.22,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10, color=NC_COLORS["text"]),
                    title_text="",
                    itemclick="toggleothers",
                    itemdoubleclick="toggle",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#24324a", size=12),
            )

            donut_chart = apply_nc_plotly_theme(
                donut_chart,
                height=chart_height + 46,
                margin=dict(l=18, r=18, t=34, b=128 if not single_product_view else 112),
                showlegend=True,
                legend_y=-0.39 if not single_product_view else -0.31,
            )
            donut_pull_values = [
                0.075 if single_product_view and str(name) != "Remaining Portfolio" else 0.012
                for name in donut_df["product_name"].astype(str).tolist()
            ]

            donut_chart.update_traces(
                marker=dict(line=dict(color="rgba(255,255,255,0.92)", width=1.6)),
                pull=donut_pull_values,
                sort=False,
            )

            donut_chart.add_annotation(
                text=get_donut_center_text(single_product_view),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                font=dict(size=12, color=NC_COLORS["axis"], family="Inter, Segoe UI, Arial"),
            )

            donut_chart.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.39 if not single_product_view else -0.31,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=9.5, color=NC_COLORS["text"]),
                    title_text="",
                    tracegroupgap=6,
                    itemsizing="constant",
                    itemclick="toggleothers",
                    itemdoubleclick="toggle",
                )
            )

            st.plotly_chart(donut_chart, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("#### Product-Level Margin Leaderboard")

        leaderboard_view = product_profitability_df[
            [
                "product_name",
                "sales",
                "profit",
                "gross_margin_pct",
                "profit_per_unit",
                "revenue_contribution_pct",
                "profit_contribution_pct",
                "margin_status",
            ]
        ].copy()

        leaderboard_view = leaderboard_view.rename(
            columns={
                "product_name": "Product",
                "sales": "Sales",
                "profit": "Gross Profit",
                "gross_margin_pct": "Gross Margin (%)",
                "profit_per_unit": "Profit per Unit",
                "revenue_contribution_pct": "Revenue Contribution (%)",
                "profit_contribution_pct": "Profit Contribution (%)",
                "margin_status": "Margin Status",
            }
        )
        
        leaderboard_row_count = len(leaderboard_view)

        if leaderboard_row_count <= 1:
            leaderboard_height = 92
        elif leaderboard_row_count <= 3:
            leaderboard_height = 52 + (leaderboard_row_count * 38)
        else:
            leaderboard_height = min(430, 52 + (leaderboard_row_count * 38))

        leaderboard_display = build_display_table(
            leaderboard_view,
            money_columns=["Sales", "Gross Profit"],
            money_2_columns=["Profit per Unit"],
            pct_columns=[
                "Gross Margin (%)",
                "Revenue Contribution (%)",
                "Profit Contribution (%)",
            ],
        )

        st.dataframe(
            leaderboard_display,
            use_container_width=True,
            hide_index=True,
            height=leaderboard_height,
            column_config=get_table_column_config(leaderboard_display),
        )

# =========================================================
# TAB 2 — DIVISION PERFORMANCE DASHBOARD
# =========================================================
with tab_division:
    st.subheader("Division Performance Dashboard")
    st.caption(
        "Revenue vs profit comparison, margin distribution, and division-level performance signals based on the selected filters."
    )

    if division_performance_df.empty:
        st.warning("Division performance data is not available for the selected filters.")
    else:
        # -----------------------------
        # Step 2B — polished division signals
        # -----------------------------
        division_performance_df = division_performance_df.copy()
        division_performance_df["margin_status"] = np.where(
            division_performance_df["gross_margin_pct"] < margin_threshold,
            "Below Threshold",
            "Healthy Margin",
        )

        top_profit_row = division_performance_df.sort_values("profit", ascending=False).iloc[0]
        top_revenue_row = division_performance_df.sort_values("sales", ascending=False).iloc[0]
        top_margin_row = division_performance_df.sort_values("gross_margin_pct", ascending=False).iloc[0]

        top_division = top_profit_row["division_name"]
        top_division_profit = top_profit_row["profit"]
        top_division_margin = top_margin_row["gross_margin_pct"]
        avg_division_margin = division_performance_df["gross_margin_pct"].mean()

        div_kpi_1, div_kpi_2, div_kpi_3, div_kpi_4 = st.columns(4)

        with div_kpi_1:
            st.metric(
                label="Divisions in View",
                value=f"{len(division_performance_df):,}",
                delta="Filtered division count",
            )

        with div_kpi_2:
            st.metric(
                label="Top Profit Division",
                value=short_label(top_division, 18),
                delta=f"{fmt_money(top_division_profit, 0)} profit",
            )

        with div_kpi_3:
            st.metric(
                label="Top Division Margin",
                value=fmt_pct(top_division_margin, 1),
                delta=f"{short_label(top_margin_row['division_name'], 16)}",
            )

        with div_kpi_4:
            st.metric(
                label="Avg Division Margin",
                value=fmt_pct(avg_division_margin, 1),
                delta="Across visible divisions",
            )

        div_chart_col_1, div_chart_col_2 = st.columns([1.15, 1])

        with div_chart_col_1:
            st.markdown("#### Revenue vs Profit by Division")

            division_melted = division_performance_df.melt(
                id_vars=["division_name"],
                value_vars=["sales", "profit"],
                var_name="metric",
                value_name="amount",
            )

            division_melted["metric"] = division_melted["metric"].replace(
                {
                    "sales": "Revenue",
                    "profit": "Gross Profit",
                }
            )

            division_melted["division_short"] = division_melted["division_name"].astype(str).apply(
                lambda x: short_label(x, 16)
            )

            revenue_profit_chart = px.bar(
                division_melted,
                x="division_short",
                y="amount",
                color="metric",
                barmode="group",
                text="amount",
                custom_data=["division_name", "metric"],
                labels={
                    "division_short": "Division",
                    "amount": "Amount",
                    "metric": "Metric",
                },
                color_discrete_map={
                    "Revenue": "#24324a",
                    "Gross Profit": "#0f8b72",
                },
            )

            revenue_profit_chart.update_traces(
                texttemplate="%{text}",
                textposition="outside",
                cliponaxis=False,
                marker_line_color="rgba(255,255,255,0.72)",
                marker_line_width=1.1,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}: <b>$%{y:,.0f}</b>"
                    "<extra></extra>"
                ),
            )

            revenue_profit_chart.update_layout(
                height=438,
                margin=dict(l=12, r=38, t=24, b=62),
                xaxis_title=None,
                yaxis_title="Amount",
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.30,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="rgba(36,50,74,0.15)",
                    font_size=12,
                    font_color="#24324a",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#24324a", size=12),
                uniformtext_minsize=9,
                uniformtext_mode="hide",
            )

            revenue_profit_chart.update_yaxes(
                gridcolor="rgba(36, 50, 74, 0.10)",
                zerolinecolor="rgba(36, 50, 74, 0.18)",
            )

            revenue_profit_chart.update_xaxes(
                tickfont=dict(size=11, color="#64748b")
            )

            revenue_profit_chart = apply_nc_plotly_theme(
                revenue_profit_chart,
                height=466,
                margin=dict(l=20, r=64, t=26, b=100),
                showlegend=True,
                legend_y=-0.34,
            )
            revenue_profit_chart.update_yaxes(tickprefix="$")

            st.plotly_chart(revenue_profit_chart, use_container_width=True, config=PLOTLY_CONFIG)

        with div_chart_col_2:
            st.markdown("#### Margin Distribution by Division")

            if (
                division_distribution_df.empty
                or "row_margin_pct" not in division_distribution_df.columns
                or not division_col
                or division_col not in division_distribution_df.columns
            ):
                st.info("Margin distribution is not available for the selected filters.")
            else:
                available_cols = [division_col, "row_margin_pct"]
                if sales_col and sales_col in division_distribution_df.columns:
                    available_cols.append(sales_col)
                if profit_col and profit_col in division_distribution_df.columns:
                    available_cols.append(profit_col)

                margin_distribution_df = division_distribution_df[available_cols].copy()
                margin_distribution_df = margin_distribution_df.replace([np.inf, -np.inf], np.nan)
                margin_distribution_df = margin_distribution_df.dropna(
                    subset=[division_col, "row_margin_pct"]
                )

                if margin_distribution_df.empty:
                    st.info("No margin distribution records available.")
                else:
                    margin_distribution_df["division_short"] = margin_distribution_df[
                        division_col
                    ].astype(str).apply(lambda x: short_label(x, 18))

                    hover_cols = {}
                    if sales_col and sales_col in margin_distribution_df.columns:
                        hover_cols[sales_col] = ":$,.0f"
                    if profit_col and profit_col in margin_distribution_df.columns:
                        hover_cols[profit_col] = ":$,.0f"

                    margin_distribution_chart = px.box(
                        margin_distribution_df,
                        x="division_short",
                        y="row_margin_pct",
                        color="division_short",
                        points="outliers",
                        labels={
                            "division_short": "Division",
                            "row_margin_pct": "Gross Margin (%)",
                        },
                        hover_data=hover_cols,
                        color_discrete_sequence=[
                            NC_COLORS["navy"],
                            NC_COLORS["teal"],
                            NC_COLORS["coral"],
                            NC_COLORS["mint"],
                            NC_COLORS["beige"],
                            NC_COLORS["navy_2"],
                        ],
                    )

                    margin_distribution_chart.update_traces(
                        boxmean=True,
                        marker=dict(size=4.5, opacity=0.58),
                        line=dict(width=1.45),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Gross Margin: <b>%{y:.1f}%</b>"
                            "<extra></extra>"
                        ),
                    )

                    y_min = max(0, margin_distribution_df["row_margin_pct"].min() - 8)
                    y_max = min(100, margin_distribution_df["row_margin_pct"].max() + 8)
                    if y_min == y_max:
                        y_min = max(0, y_min - 5)
                        y_max = min(100, y_max + 5)

                    margin_distribution_chart.update_layout(
                        height=438,
                        margin=dict(l=12, r=32, t=24, b=62),
                        xaxis_title=None,
                        yaxis_title="Gross Margin (%)",
                        showlegend=False,
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="rgba(36,50,74,0.15)",
                            font_size=12,
                            font_color="#24324a",
                        ),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#24324a", size=12),
                    )

                    margin_distribution_chart.update_yaxes(
                        range=[y_min, y_max],
                        zeroline=True,
                        zerolinecolor="rgba(36, 50, 74, 0.18)",
                        gridcolor="rgba(36, 50, 74, 0.10)",
                    )

                    margin_distribution_chart.update_xaxes(
                        tickfont=dict(size=11, color="#64748b")
                    )

                    margin_distribution_chart = apply_nc_plotly_theme(
                        margin_distribution_chart,
                        height=466,
                        margin=dict(l=20, r=48, t=26, b=82),
                        showlegend=False,
                    )

                    if len(division_distribution_df[division_col].dropna().astype(str).unique()) <= 1:
                        margin_distribution_chart.add_annotation(
                            x=0.5,
                            y=1.08,
                            xref="paper",
                            yref="paper",
                            text="Single-division view",
                            showarrow=False,
                            bgcolor="rgba(215,241,237,0.78)",
                            bordercolor="rgba(15,139,114,0.16)",
                            borderwidth=1,
                            borderpad=4,
                            font=dict(size=10, color=NC_COLORS["teal"]),
                        )

                    st.plotly_chart(margin_distribution_chart, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("#### Division Performance Summary")

        division_table = division_performance_df[
            [
                "division_name",
                "sales",
                "profit",
                "gross_margin_pct",
                "revenue_contribution_pct",
                "profit_contribution_pct",
                "margin_status",
            ]
        ].copy()

        division_table = division_table.sort_values("profit", ascending=False)

        division_table = division_table.rename(
            columns={
                "division_name": "Division",
                "sales": "Revenue",
                "profit": "Gross Profit",
                "gross_margin_pct": "Gross Margin (%)",
                "revenue_contribution_pct": "Revenue Contribution (%)",
                "profit_contribution_pct": "Profit Contribution (%)",
                "margin_status": "Margin Status",
            }
        )

        division_row_count = len(division_table)

        if division_row_count <= 1:
            division_table_height = 92
        elif division_row_count <= 3:
            division_table_height = 58 + (division_row_count * 38)
        else:
            division_table_height = min(380, 58 + (division_row_count * 38))

        division_display = build_display_table(
            division_table,
            money_columns=["Revenue", "Gross Profit"],
            pct_columns=[
                "Gross Margin (%)",
                "Revenue Contribution (%)",
                "Profit Contribution (%)",
            ],
        )

        st.dataframe(
            division_display,
            use_container_width=True,
            hide_index=True,
            height=division_table_height,
            column_config=get_table_column_config(division_display),
        )

# =========================================================
# TAB 3 — COST VS MARGIN DIAGNOSTICS
# =========================================================
with tab_cost:
    st.subheader("Cost vs Margin Diagnostics")
    st.caption(
        "Cost-sales scatter analysis, cost load visibility, and margin risk flags based on the selected filters."
    )

    if cost_diagnostics_df.empty:
        st.warning("Cost diagnostics data is not available for the selected filters.")
    else:
        cost_view_df = cost_diagnostics_df.copy()

        products_in_cost_view = len(cost_view_df)
        weighted_cost_to_sales = (
            cost_view_df["cost"].sum() / cost_view_df["sales"].sum() * 100
            if cost_view_df["sales"].sum()
            else 0
        )
        low_margin_products = int(cost_view_df["margin_status"].eq("Below Threshold").sum())

        if low_margin_products > 0:
            highest_risk_row = (
                cost_view_df[cost_view_df["margin_status"].eq("Below Threshold")]
                .sort_values(["gross_margin_pct", "cost_to_sales_pct"], ascending=[True, False])
                .iloc[0]
            )
        else:
            highest_risk_row = cost_view_df.sort_values(
                ["cost_to_sales_pct", "sales"],
                ascending=[False, False],
            ).iloc[0]

        cost_kpi_1, cost_kpi_2, cost_kpi_3, cost_kpi_4 = st.columns(4)

        with cost_kpi_1:
            st.metric(
                label="Products in View",
                value=f"{products_in_cost_view:,}",
                delta="Filtered product count",
            )

        with cost_kpi_2:
            st.metric(
                label="Cost-to-Sales Ratio",
                value=fmt_pct(weighted_cost_to_sales, 1),
                delta="Total cost ÷ total sales",
            )

        with cost_kpi_3:
            st.metric(
                label="Low Margin Products",
                value=f"{low_margin_products:,}",
                delta=f"Below {margin_threshold}%",
            )

        with cost_kpi_4:
            st.metric(
                label="Highest Cost-Risk Product",
                value=short_label(highest_risk_row["product_name"], 22),
                delta=f"{fmt_pct(highest_risk_row['gross_margin_pct'], 1)} margin",
            )

        cost_chart_col_1, cost_chart_col_2 = st.columns([1.2, 0.9])

        with cost_chart_col_1:
            st.markdown("#### Cost vs Sales Scatter Plot")

            scatter_df = cost_view_df.copy()
            scatter_df["product_short"] = scatter_df["product_name"].apply(
                lambda x: compact_product_label(x, 18)
            )
            scatter_df["plot_size"] = np.where(
                scatter_df["profit"].abs() > 0,
                scatter_df["profit"].abs(),
                scatter_df["sales"].abs(),
            )

            cost_scatter = px.scatter(
                scatter_df,
                x="sales",
                y="cost",
                color="risk_flag",
                size="plot_size",
                hover_name="product_name",
                text=None,
                labels={
                    "sales": "Sales",
                    "cost": "Cost",
                    "risk_flag": "Risk Flag",
                    "plot_size": "Profit / Sales Scale",
                    "gross_margin_pct": "Gross Margin (%)",
                    "cost_to_sales_pct": "Cost-to-Sales (%)",
                    "profit": "Gross Profit",
                },
                hover_data={
                    "sales": ":$,.0f",
                    "cost": ":$,.0f",
                    "profit": ":$,.0f",
                    "gross_margin_pct": ":.1f",
                    "cost_to_sales_pct": ":.1f",
                    "plot_size": False,
                    "product_short": False,
                },
                color_discrete_map=RISK_COLOR_MAP,
                size_max=34,
            )

            max_axis_value = max(
                float(scatter_df["sales"].max()) if not scatter_df.empty else 0,
                float(scatter_df["cost"].max()) if not scatter_df.empty else 0,
            )

            if max_axis_value > 0:
                cost_scatter.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=max_axis_value * 1.05,
                    y1=max_axis_value * 1.05,
                    line=dict(
                        color="rgba(36,50,74,0.22)",
                        width=1.4,
                        dash="dash",
                    ),
                )

            cost_scatter.update_traces(
                marker=dict(
                    line=dict(width=1.2, color="rgba(255,255,255,0.85)"),
                    opacity=0.88,
                ),
                textposition="top right",
                textfont=dict(size=9.5, color="#24324a"),
            )

            cost_scatter = add_single_point_annotation(
                cost_scatter,
                scatter_df,
                "sales",
                "cost",
                "product_name",
            )

            cost_scatter.update_layout(
                height=455,
                margin=dict(l=12, r=34, t=24, b=62),
                xaxis_title="Sales",
                yaxis_title="Cost",
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.30,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="rgba(36,50,74,0.15)",
                    font_size=12,
                    font_color="#24324a",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#24324a", size=12),
            )

            cost_scatter.update_xaxes(
                tickprefix="$",
                gridcolor="rgba(36, 50, 74, 0.10)",
                zerolinecolor="rgba(36, 50, 74, 0.18)",
            )
            cost_scatter.update_yaxes(
                tickprefix="$",
                gridcolor="rgba(36, 50, 74, 0.10)",
                zerolinecolor="rgba(36, 50, 74, 0.18)",
            )

            cost_scatter_height = get_density_chart_height(len(scatter_df), base_height=486, single_height=398)

            cost_scatter = apply_nc_plotly_theme(
                cost_scatter,
                height=cost_scatter_height,
                margin=dict(l=20, r=76, t=26, b=104),
                showlegend=True,
                legend_y=-0.34,
            )
            cost_scatter.update_xaxes(tickprefix="$")
            cost_scatter.update_yaxes(tickprefix="$")

            st.plotly_chart(cost_scatter, use_container_width=True, config=PLOTLY_CONFIG) 

        with cost_chart_col_2:
            st.markdown("#### Margin Risk Flags")

            risk_rank_df = cost_view_df.sort_values(
                ["diagnostic_priority", "gross_margin_pct", "cost_to_sales_pct"],
                ascending=[True, True, False],
            ).head(10).copy()

            risk_rank_df["product_short"] = risk_rank_df["product_name"].apply(
                lambda x: compact_product_label(x, 22)
            )
            risk_rank_df["threshold_gap"] = risk_rank_df["gross_margin_pct"] - margin_threshold

            risk_flag_chart = px.bar(
                risk_rank_df.sort_values("gross_margin_pct", ascending=True),
                x="gross_margin_pct",
                y="product_short",
                orientation="h",
                color="risk_flag",
                text="gross_margin_pct",
                labels={
                    "gross_margin_pct": "Gross Margin (%)",
                    "product_short": "Product",
                    "risk_flag": "Risk Flag",
                },
                color_discrete_map=RISK_COLOR_MAP,
            )

            risk_flag_chart.add_vline(
                x=margin_threshold,
                line_width=2,
                line_dash="dash",
                line_color="rgba(244, 135, 112, 0.85)",
                annotation_text=f"{margin_threshold}% threshold",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="#b45309",
            )

            risk_flag_chart.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                cliponaxis=False,
                marker_line_color="rgba(255,255,255,0.75)",
                marker_line_width=1,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Gross Margin: <b>%{x:.1f}%</b><br>"
                    "Risk Flag: %{customdata}"
                    "<extra></extra>"
                ),
                customdata=risk_rank_df.sort_values("gross_margin_pct", ascending=True)["risk_flag"],
            )

            risk_flag_chart.update_layout(
                height=455,
                margin=dict(l=12, r=38, t=24, b=62),
                xaxis_title="Gross Margin (%)",
                yaxis_title=None,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.30,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="rgba(36,50,74,0.15)",
                    font_size=12,
                    font_color="#24324a",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#24324a", size=12),
            )

            risk_flag_chart.update_xaxes(
                range=[0, max(100, risk_rank_df["gross_margin_pct"].max() + 8)],
                gridcolor="rgba(36, 50, 74, 0.10)",
                zerolinecolor="rgba(36, 50, 74, 0.18)",
            )

            risk_chart_height = get_density_chart_height(len(risk_rank_df), base_height=486, single_height=398)

            risk_flag_chart = apply_nc_plotly_theme(
                risk_flag_chart,
                height=risk_chart_height,
                margin=dict(l=20, r=96, t=28, b=104),
                showlegend=True,
                legend_y=-0.34,
            )

            st.plotly_chart(risk_flag_chart, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("#### Cost Diagnostic Product Table")

        cost_table = cost_view_df[
            [
                "product_name",
                "sales",
                "cost",
                "profit",
                "gross_margin_pct",
                "cost_to_sales_pct",
                "profit_per_unit",
                "risk_flag",
                "margin_status",
                "diagnostic_priority",
            ]
        ].copy()

        cost_table = cost_table.sort_values(
            ["diagnostic_priority", "gross_margin_pct", "cost_to_sales_pct"],
            ascending=[True, True, False],
        ).drop(columns=["diagnostic_priority"])

        cost_table = cost_table.rename(
            columns={
                "product_name": "Product",
                "sales": "Sales",
                "cost": "Cost",
                "profit": "Gross Profit",
                "gross_margin_pct": "Gross Margin (%)",
                "cost_to_sales_pct": "Cost-to-Sales (%)",
                "profit_per_unit": "Profit per Unit",
                "risk_flag": "Risk Flag",
                "margin_status": "Margin Status",
            }
        )

        cost_row_count = len(cost_table)

        if cost_row_count <= 1:
            cost_table_height = 92
        elif cost_row_count <= 5:
            cost_table_height = 58 + (cost_row_count * 38)
        else:
            cost_table_height = min(420, 58 + (cost_row_count * 38))

        cost_display = build_display_table(
            cost_table,
            money_columns=["Sales", "Cost", "Gross Profit"],
            money_2_columns=["Profit per Unit"],
            pct_columns=["Gross Margin (%)", "Cost-to-Sales (%)"],
        )

        st.dataframe(
            cost_display,
            use_container_width=True,
            hide_index=True,
            height=cost_table_height,
            column_config=get_table_column_config(cost_display),
        )

# =========================================================
# TAB 4 — PROFIT CONCENTRATION ANALYSIS
# =========================================================
with tab_pareto:
    st.subheader("Profit Concentration Analysis")
    st.caption(
        "Pareto profit concentration, dependency indicators, and product-level contribution risk based on the selected filters."
    )

    if profit_concentration_df.empty:
        st.warning("Profit concentration data is not available for the selected filters.")
    else:
        concentration_df = profit_concentration_df.copy()

        products_in_concentration_view = len(concentration_df)
        total_concentration_profit = concentration_df["profit"].sum()
        top_1_profit_share = (
            concentration_df.head(1)["profit"].sum() / total_concentration_profit * 100
            if total_concentration_profit
            else 0
        )
        top_3_profit_share = (
            concentration_df.head(3)["profit"].sum() / total_concentration_profit * 100
            if total_concentration_profit
            else 0
        )
        top_5_profit_share = (
            concentration_df.head(5)["profit"].sum() / total_concentration_profit * 100
            if total_concentration_profit
            else 0
        )

        core_80_count = int(concentration_df["pareto_tier"].eq("Core 80% Driver").sum())

        if top_1_profit_share >= 35 or core_80_count <= 2:
            dependency_risk_level = "High Dependency"
        elif top_3_profit_share >= 70 or core_80_count <= 4:
            dependency_risk_level = "Moderate Dependency"
        else:
            dependency_risk_level = "Balanced Portfolio"

        pareto_kpi_1, pareto_kpi_2, pareto_kpi_3, pareto_kpi_4 = st.columns(4)

        with pareto_kpi_1:
            st.metric(
                label="Products in View",
                value=f"{products_in_concentration_view:,}",
                delta="Positive-profit products",
            )

        with pareto_kpi_2:
            st.metric(
                label="Top Product Profit Share",
                value=fmt_pct(top_1_profit_share, 1),
                delta=short_label(concentration_df.iloc[0]["product_name"], 22),
            )

        with pareto_kpi_3:
            st.metric(
                label="Products Driving 80% Profit",
                value=f"{core_80_count:,}",
                delta="Pareto core drivers",
            )

        with pareto_kpi_4:
            st.metric(
                label="Dependency Risk Level",
                value=dependency_risk_level,
                delta=f"Top 3: {fmt_pct(top_3_profit_share, 1)}",
            )

        pareto_chart_col_1, pareto_chart_col_2 = st.columns([1.25, 0.85])

        with pareto_chart_col_1:
            st.markdown("#### Pareto Profit Concentration")

            pareto_chart_df = concentration_df.head(10).copy()

            # STEP 7C-v2: safer Pareto labels.
            # Keep full product names in hover, but use short one-line axis labels to prevent overlap.
            pareto_chart_df["product_short"] = pareto_chart_df["product_name"].apply(
                lambda x: compact_product_label(x, 16).replace("<br>", " ")
            )

            for pareto_numeric_col in [
                "profit",
                "profit_contribution_pct",
                "cumulative_profit_pct",
                "gross_margin_pct",
            ]:
                if pareto_numeric_col in pareto_chart_df.columns:
                    pareto_chart_df[pareto_numeric_col] = (
                        pd.to_numeric(pareto_chart_df[pareto_numeric_col], errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0)
                    )

            pareto_chart_df["profit_label"] = pareto_chart_df["profit"].apply(fmt_money_compact)

            pareto_chart = px.bar(
                pareto_chart_df,
                x="product_short",
                y="profit",
                color="pareto_tier",
                text="profit_label",
                custom_data=[
                    "product_name",
                    "profit_contribution_pct",
                    "cumulative_profit_pct",
                    "gross_margin_pct",
                ],
                labels={
                    "product_short": "Product",
                    "profit": "Gross Profit",
                    "pareto_tier": "Pareto Tier",
                },
                color_discrete_map={
                    "Core 80% Driver": NC_COLORS["navy"],
                    "Long-Tail Profit": NC_COLORS["mint"],
                },
            )

            pareto_chart.add_scatter(
                x=pareto_chart_df["product_short"],
                y=pareto_chart_df["cumulative_profit_pct"],
                mode="lines+markers",
                name="Cumulative Profit %",
                yaxis="y2",
                line=dict(color="#f48770", width=3),
                marker=dict(size=8, color="#f48770"),
                customdata=pareto_chart_df["product_name"],
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Cumulative Profit: <b>%{y:.1f}%</b>"
                    "<extra></extra>"
                ),
            )

            pareto_chart.add_hline(
                y=80,
                line_dash="dash",
                line_color="rgba(15,139,114,0.50)",
                line_width=1.6,
                annotation_text="80% benchmark",
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#0f766e",
                yref="y2",
            )

            pareto_chart.update_traces(
                selector=dict(type="bar"),
                # STEP 7C-v2: profit_label is already formatted by fmt_money_compact.
                # Using a numeric format here caused $NaN labels on the chart.
                texttemplate="%{text}",
                textposition="outside",
                textfont=dict(size=10, color=NC_COLORS["text"], family="Inter, Segoe UI, Arial"),
                cliponaxis=False,
                marker_line_color="rgba(255,255,255,0.72)",
                marker_line_width=1.1,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Gross Profit: <b>$%{y:,.0f}</b><br>"
                    "Profit Share: %{customdata[1]:.1f}%<br>"
                    "Cumulative Profit: %{customdata[2]:.1f}%<br>"
                    "Gross Margin: %{customdata[3]:.1f}%"
                    "<extra></extra>"
                ),
            )

            pareto_chart.update_layout(
                height=470,
                margin=dict(l=12, r=42, t=24, b=92),
                xaxis_title=None,
                yaxis_title="Gross Profit",
                yaxis2=dict(
                    title="Cumulative Profit %",
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                    showgrid=False,
                    tickformat=".0f",
                ),
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.36,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="rgba(36,50,74,0.15)",
                    font_size=12,
                    font_color="#24324a",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#24324a", size=12),
                uniformtext_minsize=8,
                uniformtext_mode="hide",
            )

            pareto_chart.update_xaxes(
                tickangle=-18,
                tickfont=dict(size=9.2, color="#64748b"),
                automargin=True,
            )
            pareto_chart.update_yaxes(
                tickprefix="$",
                gridcolor="rgba(36, 50, 74, 0.10)",
                zerolinecolor="rgba(36, 50, 74, 0.18)",
            )

            pareto_chart_height = get_density_chart_height(len(pareto_chart_df), base_height=520, single_height=420)

            pareto_chart = apply_nc_plotly_theme(
                pareto_chart,
                height=pareto_chart_height,
                margin=dict(l=24, r=82, t=34, b=124),
                showlegend=True,
                legend_y=-0.38,
            )
            pareto_chart.update_yaxes(tickprefix="$", secondary_y=False)
            pareto_chart.update_layout(
                yaxis2=dict(
                    title=dict(
                        text="Cumulative Profit %",
                        font=dict(color=NC_COLORS["axis"], size=11),
                    ),
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                    showgrid=False,
                    tickformat=".0f",
                    tickfont=dict(color=NC_COLORS["axis"], size=10),
                )
            )

            if len(pareto_chart_df) <= 1:
                pareto_chart.add_annotation(
                    x=0.5,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text="Single-product concentration view",
                    showarrow=False,
                    bgcolor="rgba(215,241,237,0.78)",
                    bordercolor="rgba(15,139,114,0.16)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=10, color=NC_COLORS["teal"]),
                )

            st.plotly_chart(pareto_chart, use_container_width=True, config=PLOTLY_CONFIG)

        with pareto_chart_col_2:
            st.markdown("#### Dependency Indicators")

            dependency_df = pd.DataFrame(
                [
                    {
                        "indicator": "Top 1 Product",
                        "profit_share_pct": top_1_profit_share,
                        "status": "High" if top_1_profit_share >= 35 else "Stable",
                    },
                    {
                        "indicator": "Top 3 Products",
                        "profit_share_pct": top_3_profit_share,
                        "status": "High" if top_3_profit_share >= 70 else "Stable",
                    },
                    {
                        "indicator": "Top 5 Products",
                        "profit_share_pct": top_5_profit_share,
                        "status": "High" if top_5_profit_share >= 85 else "Stable",
                    },
                    {
                        "indicator": "Core 80% Drivers",
                        "profit_share_pct": 80,
                        "status": dependency_risk_level,
                    },
                ]
            )

            dependency_chart = px.bar(
                dependency_df,
                x="profit_share_pct",
                y="indicator",
                orientation="h",
                color="status",
                text="profit_share_pct",
                labels={
                    "profit_share_pct": "Profit Share (%)",
                    "indicator": "Dependency Indicator",
                    "status": "Risk Status",
                },
                color_discrete_map=DEPENDENCY_COLOR_MAP,
            )

            dependency_chart.add_vline(
                x=80,
                line_dash="dash",
                line_color="rgba(36,50,74,0.35)",
                line_width=1.5,
                annotation_text="80%",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_font_color="#24324a",
            )

            dependency_chart.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                cliponaxis=False,
                marker_line_color="rgba(255,255,255,0.75)",
                marker_line_width=1.1,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Profit Share: <b>%{x:.1f}%</b><br>"
                    "Status: %{customdata}"
                    "<extra></extra>"
                ),
                customdata=dependency_df["status"],
            )

            dependency_chart.update_layout(
                height=470,
                margin=dict(l=12, r=42, t=24, b=72),
                xaxis_title="Profit Share (%)",
                yaxis_title=None,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.28,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    bordercolor="rgba(36,50,74,0.15)",
                    font_size=12,
                    font_color="#24324a",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#24324a", size=12),
            )

            dependency_chart.update_xaxes(
                range=[0, 105],
                ticksuffix="%",
                gridcolor="rgba(36, 50, 74, 0.10)",
                zerolinecolor="rgba(36, 50, 74, 0.18)",
            )

            dependency_chart_height = get_density_chart_height(
                len(concentration_df),
                base_height=506,
                single_height=420,
            )

            dependency_chart = apply_nc_plotly_theme(
                dependency_chart,
                height=dependency_chart_height,
                margin=dict(l=20, r=102, t=28, b=104),
                showlegend=True,
                legend_y=-0.34,
            )

            if len(concentration_df) <= 1:
                dependency_chart.update_traces(opacity=0.82)
                dependency_chart.add_annotation(
                    x=0.5,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text="Single product = naturally high dependency",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.86)",
                    bordercolor="rgba(36,50,74,0.12)",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=10, color=NC_COLORS["text"]),
                )

            st.plotly_chart(dependency_chart, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("#### Profit Concentration Product Table")

        concentration_table = concentration_df[
            [
                "rank",
                "product_name",
                "profit",
                "profit_contribution_pct",
                "cumulative_profit_pct",
                "gross_margin_pct",
                "pareto_tier",
                "dependency_status",
                "margin_status",
            ]
        ].copy()

        concentration_table = concentration_table.rename(
            columns={
                "rank": "Rank",
                "product_name": "Product",
                "profit": "Gross Profit",
                "profit_contribution_pct": "Profit Contribution (%)",
                "cumulative_profit_pct": "Cumulative Profit (%)",
                "gross_margin_pct": "Gross Margin (%)",
                "pareto_tier": "Pareto Tier",
                "dependency_status": "Dependency Status",
                "margin_status": "Margin Status",
            }
        )

        concentration_row_count = len(concentration_table)

        if concentration_row_count <= 1:
            concentration_table_height = 92
        elif concentration_row_count <= 5:
            concentration_table_height = 58 + (concentration_row_count * 38)
        else:
            concentration_table_height = min(440, 58 + (concentration_row_count * 38))

        concentration_display = build_display_table(
            concentration_table,
            money_columns=["Gross Profit"],
            pct_columns=[
                "Profit Contribution (%)",
                "Cumulative Profit (%)",
                "Gross Margin (%)",
            ],
            int_columns=["Rank"],
        )

        st.dataframe(
            concentration_display,
            use_container_width=True,
            hide_index=True,
            height=concentration_table_height,
            column_config=get_table_column_config(concentration_display),
        )

# =========================================================
# END OF STEP 4A
# =========================================================

# -------------------------
# FOOTER SECTION
# -------------------------
st.markdown("---")

st.markdown("### 📌 Project Information & Credits")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
**👨‍💻 Developed by:** Mohit Gupta  
**🎯 Role:** Data Analyst Intern
        """
    )

with c2:
    st.markdown(
        """
**📊 Project:** Nassau Candy Product Line Profitability & Margin Performance Analysis Dashboard  
**🏢 Organization:** Unified Mentor Pvt. Ltd.
        """
    )

with c3:
    st.markdown(
        """
**👨‍🏫 Mentor:** Saiprasad Kagne  
**📅 Year:** 2026
        """
    )

st.markdown(
    """
<div style="
    text-align: center;
    margin-top: 10px;
    color: #6b563d;
    font-size: 14px;
    font-weight: 600;
">
    💡 Built using Python, Pandas, Plotly & Streamlit
</div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)