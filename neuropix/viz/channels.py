from typing import Dict, Optional, Tuple, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import colorsys

def _generate_palette(scheme: str, n_colors: int) -> List:
        """Generate a color palette with improved contrast."""
        if scheme == 'rainbow':
            return [colorsys.hsv_to_rgb(i/n_colors, 0.8, 0.9) for i in range(n_colors)]
        return sns.color_palette(scheme, n_colors)

@dataclass
class PlotConfig:
    """Configuration class for plot styling."""
    figsize: Tuple[int, int] = (15, 8)
    dpi: int = 100
    title_fontsize: int = 16
    label_fontsize: int = 12
    tick_fontsize: int = 10
    line_width: float = 1.2
    grid: bool = True
    grid_style: str = '--'
    grid_alpha: float = 0.7
    legend_position: str = 'right'  # 'right', 'bottom', or 'inside'
    style = 'whitegrid'
    context = 'notebook'
    color_scheme = 'husl'
    n_colors = 20
    color_palette = _generate_palette('husl', 20)

def _setup_legend(ax: plt.Axes, config: PlotConfig):
    """Configure legend position and style."""
    if config.legend_position == 'right':
        ax.legend(bbox_to_anchor=(1.05, 1), 
                    loc='upper left',
                    borderaxespad=0.,
                    fontsize=config.label_fontsize-2)
    elif config.legend_position == 'bottom':
        ax.legend(bbox_to_anchor=(0.5, -0.15),
                    loc='upper center',
                    borderaxespad=0.,
                    fontsize=config.label_fontsize-2,
                    ncol=3)
    else:  # 'inside'
        ax.legend(loc='best', fontsize=config.label_fontsize-2)


def plot_channels(data: Dict,
                     time_period_name: str,
                     window: int,
                     config: Optional[PlotConfig] = None,
                     conv_to_uv: bool = True,
                     ylim: Optional[tuple] = None,
                     highlight_channels: Optional[List[str]] = None,
                     show_events: Optional[Dict[str, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot individual channel data with enhanced visualization options.
        
        Args:
            data: Dictionary containing time and channel data
            time_period_name: Name of the time period for the title
            config: PlotConfig object for customizing the plot
            conv_to_uv: Convert values to microvolts
            ylim: Y-axis limits (min, max)
            highlight_channels: List of channel names to highlight
            show_events: Dictionary of event names and their timestamps
            
        Returns:
            Tuple of (Figure, Axes) objects
        """
        
        config = config or PlotConfig()
        sns.set_style(config.style)
        sns.set_context(config.context)
        plt.rcParams.update({
            # 'font.family': 'sans-serif',
            # 'font.sans-serif': ['Arial'],
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False
        })

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        # Plot channels
        for idx, channel_data in enumerate(data):
            channel_name = time_period_name[idx][0]    
            is_highlighted = highlight_channels and channel_name in highlight_channels
            line_width = config.line_width * (1.5 if is_highlighted else 1.0)
            alpha = 1.0 if is_highlighted else 0.7
            
            ax.plot(time_period_name[idx], 
                   channel_data.transpose(), 
                   label=f'Channel {channel_name}',
                   color=config.color_palette[idx % len(config.color_palette)],
                   linewidth=line_width,
                   alpha=alpha)
        
        # Add events if specified
        if show_events:
            for event_name, timestamp in show_events.items():
                ax.axvline(x=timestamp/1000, color='red', linestyle='--', alpha=0.5)
                ax.text(timestamp/1000, ax.get_ylim()[1], event_name, 
                       rotation=90, va='bottom')
        
        # Customize appearance
        ax.set_title(f'Time Period of Neural Activity: from {time_period_name[idx][0]}s to {int(time_period_name[idx][0])+window}s', 
                    fontsize=config.title_fontsize, 
                    pad=20)
        
        ax.set_xlabel('Time (s)', fontsize=config.label_fontsize)
        ax.set_ylabel('Voltage (µV)' if conv_to_uv else 'Raw value', 
                     fontsize=config.label_fontsize)
        
        ax.tick_params(axis='both', labelsize=config.tick_fontsize)
        
        if ylim:
            ax.set_ylim(ylim)
            
        if config.grid:
            ax.grid(True, linestyle=config.grid_style, alpha=config.grid_alpha)
            
        _setup_legend(ax, config)
        plt.tight_layout()
        
        return fig, ax