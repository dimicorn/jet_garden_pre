import os
from typing import Optional
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from .load import fits2numpy
from .preprocess import preprocess, preprocess_lognorm


CMAP = "gray"
TITLE_FONTSIZE = 10
XLABEL_FONTSIZE = 8


class Visualize:
    def __init__(self, fits_path: str) -> None:
        """_summary_

        Args:
            fits_path (str): _description_
        """
        self.fits_path = fits_path
        self.data_raw = fits2numpy(fits_path)
        self.data_preprocessed = preprocess(self.data_raw)
        self.data_lognorm = preprocess_lognorm(self.data_raw)

    @property
    def xlabel_raw(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self._get_xlabel(self.data_raw)

    @property
    def xlabel_preprocessed(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self._get_xlabel(self.data_preprocessed)

    @property
    def xlabel_lognorm(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self._get_xlabel(self.data_lognorm)

    @staticmethod
    def _get_xlabel(im: np.ndarray) -> str:
        """_summary_

        Args:
            im (np.ndarray): _description_

        Returns:
            str: _description_
        """
        return (
            f"Max: {im.max():.1f}, Min: {im.min():.1f}, Sum: {im.sum():.1f}\n"
            f"Height: {im.shape[0]}, Width: {im.shape[1]}"
        )

    @staticmethod
    def draw_from_data(
        ax: Axes,
        data: np.ndarray,
        title: Optional[str],
        xlabel: Optional[str],
    ) -> Axes:
        """_summary_

        Args:
            ax (Axes): _description_
            data (np.ndarray): _description_
            title (Optional[str]): _description_
            xlabel (Optional[str]): _description_

        Returns:
            Axes: _description_
        """
        # TODO: horizontal flip
        ax.imshow(data, cmap=CMAP)
        if title is not None:
            ax.set_title(title, fontsize=TITLE_FONTSIZE)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=XLABEL_FONTSIZE)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    @staticmethod
    def draw(
        data: np.ndarray,
        savefig: bool = False,
        img_name: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """_summary_

        Args:
            data (np.ndarray): _description_
            savefig (bool, optional): _description_. Defaults to False.
            img_name (Optional[str], optional): _description_. Defaults to None.
            title (Optional[str], optional): _description_. Defaults to None.
            xlabel (Optional[str], optional): _description_. Defaults to None.
            ax (Optional[Axes], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            Axes: _description_
        """
        owns_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            owns_fig = True
        else:
            fig = ax.figure

        ax = Visualize.draw_from_data(ax, data, title=title, xlabel=xlabel)

        if owns_fig:
            plt.tight_layout()
            if savefig:
                if img_name is None:
                    raise ValueError("img_name must be provided when savefig=True")
                fig.savefig(img_name)
                plt.close(fig)
        return ax

    @staticmethod
    def draw_raw(
        fits_path: str,
        savefig: bool = False,
        img_name: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """_summary_

        Args:
            fits_path (str): _description_
            savefig (bool, optional): _description_. Defaults to False.
            img_name (Optional[str], optional): _description_. Defaults to None.
            title (Optional[str], optional): _description_. Defaults to None.
            xlabel (Optional[str], optional): _description_. Defaults to None.
            ax (Optional[Axes], optional): _description_. Defaults to None.

        Returns:
            Axes: _description_
        """
        data = fits2numpy(fits_path)
        return Visualize.draw(data, savefig, img_name, title, xlabel, ax)

    @staticmethod
    def draw_preprocessed(
        fits_path: str,
        savefig: bool = False,
        img_name: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """_summary_

        Args:
            fits_path (str): _description_
            savefig (bool, optional): _description_. Defaults to False.
            img_name (Optional[str], optional): _description_. Defaults to None.
            title (Optional[str], optional): _description_. Defaults to None.
            xlabel (Optional[str], optional): _description_. Defaults to None.
            ax (Optional[Axes], optional): _description_. Defaults to None.

        Returns:
            Axes: _description_
        """
        data = preprocess(fits2numpy(fits_path))
        return Visualize.draw(data, savefig, img_name, title, xlabel, ax)

    @staticmethod
    def draw_lognorm(
        fits_path: str,
        savefig: bool = False,
        img_name: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """_summary_

        Args:
            fits_path (str): _description_
            savefig (bool, optional): _description_. Defaults to False.
            img_name (Optional[str], optional): _description_. Defaults to None.
            title (Optional[str], optional): _description_. Defaults to None.
            xlabel (Optional[str], optional): _description_. Defaults to None.
            ax (Optional[Axes], optional): _description_. Defaults to None.

        Returns:
            Axes: _description_
        """
        data = preprocess_lognorm(fits2numpy(fits_path))
        return Visualize.draw(data, savefig, img_name, title, xlabel, ax)

    def save_npy(
        self,
        npy_path: str,
        npy_log_path: str,
        filename: Optional[str] = None,
        create_dirs: bool = True,
    ) -> None:
        """_summary_

        Args:
            npy_path (str): _description_
            npy_log_path (str): _description_
            filename (Optional[str], optional): _description_. Defaults to None.
            create_dirs (bool, optional): _description_. Defaults to True.
        """
        if filename is None:
            filename = os.path.splitext(os.path.basename(self.fits_path))[0]

        if create_dirs:
            os.makedirs(npy_path, exist_ok=True)
            os.makedirs(npy_log_path, exist_ok=True)

        np.save(os.path.join(npy_path, filename), self.data_preprocessed)
        np.save(os.path.join(npy_log_path, f"{filename}_lognorm"), self.data_lognorm)

    def draw_all(
        self,
        img_dir: str,
        savefig: bool = False,
        img_name: Optional[str] = None,
        suptitle: Optional[str] = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """_summary_

        Args:
            img_dir (str): _description_
            savefig (bool, optional): _description_. Defaults to False.
            img_name (Optional[str], optional): _description_. Defaults to None.
            suptitle (Optional[str], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            tuple[plt.Figure, np.ndarray]: _description_
        """
        os.makedirs(img_dir, exist_ok=True)
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))

        if suptitle is not None:
            fig.suptitle(suptitle)

        axs[0] = Visualize.draw(
            self.data_raw,
            title="Raw map",
            xlabel=self.xlabel_raw,
            ax=axs[0],
        )
        axs[1] = Visualize.draw(
            self.data_preprocessed,
            title="Preprocessed map",
            xlabel=self.xlabel_preprocessed,
            ax=axs[1],
        )
        axs[2] = Visualize.draw(
            self.data_lognorm,
            title="Lognorm map",
            xlabel=self.xlabel_lognorm,
            ax=axs[2],
        )

        axs[3].hist(self.data_lognorm.ravel(), bins=100)
        axs[3].set_title("Histogram of log-normalized values")
        axs[3].set_yscale("log")
        axs[3].grid(True)
        axs[3].set_box_aspect(1)
        plt.tight_layout()

        if savefig:
            if img_name is None:
                raise ValueError("img_name must be provided when savefig=True")
            fig.savefig(os.path.join(img_dir, img_name))
            plt.close(fig)
        return fig, axs
