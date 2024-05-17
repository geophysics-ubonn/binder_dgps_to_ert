#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import zipfile

import ipywidgets as widgets
import shapely
import shapely.plotting

# from ipywidgets import GridspecLayout
from ipywidgets import GridBox, Layout

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import geojson
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest

from pyproj import CRS

from pyproj import Transformer

import crtomo
import tempfile
import subprocess

from IPython.display import display


# class ert_geojson(object):
#     """
#     Import electrode coordinates from a geojson file
#     """

#     def __init__(self, geojson_file):
#         # assert os.path.isfile(geojson_file)
#         self.filename = geojson_file
#         self.wgs84_points = []
#         self.utm_coords = None

#         self.load_geojson_file()
#         # self.summarize()

#         # self.output = widgets.Output()

#     def _prepare_utm_transformer(self):

#         print('POINTS', self.wgs84_points)

#         # First we need to get the UTM zone for our data points
#         lon_min = self.wgs84_points[:, 0].min()
#         lon_max = self.wgs84_points[:, 0].max()
#         # xdiff = xmax - xmin

#         lat_min = self.wgs84_points[:, 1].min()
#         lat_max = self.wgs84_points[:, 1].max()

#         print(lat_min, lat_max, lon_min, lon_max)

#         utm_crs_list = query_utm_crs_info(
#             datum_name="WGS 84",
#             area_of_interest=AreaOfInterest(
#                 west_lon_degree=lon_min,
#                 south_lat_degree=lat_min,
#                 east_lon_degree=lon_max,
#                 north_lat_degree=lat_max,
#             ),
#         )
#         print('CODE', utm_crs_list[0].code)
#         self.utm_crs = CRS.from_epsg(utm_crs_list[0].code)
#         print(self.utm_crs.to_wkt(pretty=True))

#         # From SW Maps Homepage: All data recorded and exported by SW Maps is
#         # in the WGS84 geographic
#         # coordinate system (EPSG:4326).
#         self.wgs84_crs = CRS.from_string("EPSG:4326")

#         self.transformer = Transformer.from_crs(self.wgs84_crs, self.utm_crs)

#     def _to_utm(self):
#         if self.utm_coords is not None:
#             # work was already done
#             return
#         self._prepare_utm_transformer()
#         utm_coords = []
#         for point in self.wgs84_points:
#             utm_tmp = self.transformer.transform(point[1], point[0])
#             utm_coords += [[utm_tmp[0], utm_tmp[1], point[2]], ]

#         self.utm_coords = np.array(utm_coords)

#         coords = self.utm_coords.copy()
#         self.xy_distances_abs = np.hstack((
#             0,
#             np.sqrt(
#                 (coords[1:, 0] - coords[0:-1, 0]) ** 2 +
#                 (coords[1:, 1] - coords[0:-1, 1]) ** 2
#             )
#         ))

#         for i in range(3):
#             coords[:, i] -= coords[:, i].min()
#         self.xy_distances_rel = np.hstack((
#             0,
#             np.sqrt(
#                 (coords[1:, 0] - coords[0:-1, 0]) ** 2 +
#                 (coords[1:, 1] - coords[0:-1, 1]) ** 2
#             )
#         ))
#         print('Distances:')
#         print(self.xy_distances_abs)
#         print(self.xy_distances_rel)

#     def load_geojson_file(self):
#         if isinstance(self.filename, io.BytesIO):
#             self.filename.seek(0)
#             self.data = geojson.load(self.filename)
#         else:
#             with open(self.filename, 'r') as fid:
#                 self.data = geojson.load(fid)

#         self.wgs84_points = np.array(
#             list(
#                 geojson.utils.coords(
#                     self.data['features']
#                 )
#             )
#         )
#         self._to_utm()

#     def summarize(self):
#         print('Data filename:', self.filename)
#         print('Number of data points:', self.wgs84_points.shape)
#         print(self.wgs84_points)

#     def plot_to_map(self):
#         imagery = OSM(
#             cache=True,
#         )
#         fig = plt.figure(figsize=(20, 20))
#         ax = fig.add_subplot(
#             1, 1, 1, projection=imagery.crs
#         )

#         # try to find some sane extents based on the loaded data points
#         # longitude (east-west, x-axis)
#         xmin = self.wgs84_points[:, 0].min()
#         xmax = self.wgs84_points[:, 0].max()
#         xdiff = xmax - xmin

#         ymin = self.wgs84_points[:, 1].min()
#         ymax = self.wgs84_points[:, 1].max()
#         ydiff = ymax - ymin

#         extent = [
#             self.wgs84_points[:, 0].min() - xdiff / 2,
#             self.wgs84_points[:, 0].max() + xdiff / 2,
#             self.wgs84_points[:, 1].min() - ydiff / 2,
#             self.wgs84_points[:, 1].max() + ydiff / 2,
#         ]
#         ax.set_extent(extent, ccrs.PlateCarree())
#         # ax.set_extent(extent, ccrs.UTM(32))
#         # [
#         # Spiekeroog
#         # 7.6587,
#         # 7.84,
#         # 53.75,
#         # 53.79R

#         # ], ccrs.PlateCarree()
#         # )

#         # Add the imagery to the map.
#         detail = 19
#         ax.add_image(imagery, detail)

#         # skip the last character, which we implicitly assume to be N
#         utm_zone = int(self.utm_crs.utm_zone[:-1])
#         utm_projection = ccrs.UTM(utm_zone)

#         for index, point in enumerate(self.utm_coords):
#             ax.scatter(
#                 point[0],
#                 point[1],
#                 transform=utm_projection,
#                 s=100,
#                 color='r',
#             )
#             ax.scatter(
#                 point[1],
#                 point[0],
#                 transform=utm_projection,
#                 s=100,
#                 color='g',
#             )

#         for index, point in enumerate(self.wgs84_points):
#             # ax.scatter(
#             #     point[0],
#             #     point[1],
#             #     transform=ccrs.PlateCarree(),
#             #     s=100,
#             #     color='k',
#             # )
#             ax.annotate(
#                 '{}'.format(int(index)),
#                 xy=(point[0], point[1]),
#                 transform=ccrs.PlateCarree(),
#                 color='r',
#                 fontsize=26,
#                 bbox=dict(boxstyle="round", fc="0.8"),
#                 textcoords='offset pixels',
#                 xytext=(10, 25),
#             )

#         pstart = self.wgs84_points[0]
#         pend = self.wgs84_points[-1]

#         ax.plot(
#             [pstart[0], pend[0]],
#             [pstart[1], pend[1]],
#             '-',
#             transform=ccrs.PlateCarree(),
#         )

#         return fig, ax

#     def plot_topography(self, relative=False):
#         coords = self.utm_coords.copy()
#         if relative:
#             for i in range(3):
#                 coords[:, i] -= coords[:, i].min()
#             xy_distances = self.xy_distances_rel
#         else:
#             xy_distances = self.xy_distances_abs

#         fig, axes = plt.subplots(1, 2)
#         ax = axes[0]
#         ax.plot(
#             np.cumsum(xy_distances),
#             coords[:, 2],
#             '.-',
#         )
#         ax.grid()
#         ax.set_xlabel('distance [m]')
#         ax.set_ylabel('height [m]')
#         if relative:
#             ax.set_title('Relative heights', loc='left', fontsize=7)
#         ax.set_title('Topography', loc='right')

#         ax = axes[1]
#         ax.plot(
#             range(1, xy_distances.size),
#             xy_distances[1:],
#             '.-',
#         )
#         ax.grid()
#         ax.set_ylabel('Distances to prev. el. [m]')
#         ax.set_xlabel('Electrode Nr')
#         return fig, ax


class electrode_manager(object):
    def __init__(self, electrode_positions, output=None):
        self.el_coords_orig = electrode_positions
        self.electrode_positions = np.hstack((
            self.el_coords_orig[:, :],
            np.ones(electrode_positions.shape[0])[:, np.newaxis],
        ))
        # debug
        # self.electrode_positions[1, 3] = 0
        self.vbox = None

        self.output = output

    def set_status_use_as_electrode(self, index, change):
        self.electrode_positions[index, 3] = int(change['new'])
        self._update_widgets()

    def _build_widgets(self):
        el_widgets = []
        for index, electrode in enumerate(self.electrode_positions):
            items = []
            items += [
                widgets.Label('_'),
                widgets.Button(description='Move down'),
                widgets.Button(description='Move up'),
                widgets.Checkbox(
                    value=True,
                    description='Use as electrode',
                    disabled=False,
                    indent=False
                ),
            ]

            items[1].on_click(
                lambda x, eindex=index: self.move_down(x, eindex))
            items[2].on_click(
                lambda x, eindex=index: self.move_up(x, eindex))

            items[3].observe(
                lambda change, eindex=index: self.set_status_use_as_electrode(
                    eindex, change), names='value'
            )
            hbox = widgets.HBox(items)
            el_widgets += [hbox]

        vbox = widgets.VBox(el_widgets)
        self.vbox = vbox
        self._update_widgets()

    def _update_widgets(self):
        active_electrode_index = 0

        for index, electrode in enumerate(self.electrode_positions):
            line = self.vbox.children[index].children
            if electrode[3] == 0:
                line[0].value = 'Electrode -: ({}/{}/{})'.format(*electrode)
                # move down button
                line[1].disabled = True
                # move up button
                line[2].disabled = True
                # use-as-electrode checkbox
                line[3].value = False
            else:
                line[0].value = 'Electrode {}: ({}/{}/{})'.format(
                    active_electrode_index, *electrode
                )
                # move-down button
                line[1].disabled = False
                # move up button
                line[2].disabled = False
                # use-as-electrode checkbox
                line[3].value = True
                active_electrode_index += 1

    def set_active_state(self, index, state):
        pass

    def move_down(self, button, index):
        print('Moving down {} -> {}'.format(index, index + 1))
        new_position = index + 1
        if new_position >= self.electrode_positions.shape[0]:
            print('doing nothing')
            return
        self.electrode_positions = np.vstack((
            self.electrode_positions[0:index, :],
            self.electrode_positions[index + 1, :],
            self.electrode_positions[index, :],
            self.electrode_positions[index + 2:, :],
        ))
        print(self.electrode_positions)
        self._update_widgets()
        self.show()

    def move_up(self, button, index):
        print('Moving up {} -> {}'.format(index, index - 1))
        new_position = index - 1
        if new_position < 0:
            print('doing nothing')
            return
        self.electrode_positions = np.vstack((
            self.electrode_positions[0:index - 1, :],
            self.electrode_positions[index, :],
            self.electrode_positions[index - 1, :],
            self.electrode_positions[index + 1:, :],
        ))
        self._update_widgets()
        self.show()

    def show(self):
        if self.vbox is None:
            self._build_widgets()

        if self.output is not None:
            self.output.clear_output()
            with self.output:
                display(self.vbox)
        else:
            display(self.vbox)

    def get_electrode_positions(self):
        indices = np.where(self.electrode_positions[:, 3])
        return np.vstack((
            self.electrode_positions[indices, 0],
            self.electrode_positions[indices, 2],
        )).T


class crtomo_mesh(object):
    def __init__(self, electrode_positions, output=None):
        self.electrode_positions = electrode_positions

        # compute a few parameters
        self.distance_x = np.abs(
            self.electrode_positions[:, 0].max(
            ) - self.electrode_positions[:, 0].min()
        )
        self.distance_z = np.abs(
            self.electrode_positions[:, 1].max(
            ) - self.electrode_positions[:, 1].min()
        )
        self.distances = np.sqrt(
            (self.electrode_positions[1:, 0] -
             self.electrode_positions[:-1, 0]) ** 2 +
            (self.electrode_positions[1:, 1] -
             self.electrode_positions[:-1, 1]) ** 2
        )

        self.eldist_min = self.distances.min()
        self.eldist_max = self.distances.max()

        self.mesh = None
        self.gui = []

        if output is None:
            self.output = widgets.Output()
        else:
            self.output = output

        # used to plot the geometry
        self.output_geometry = widgets.Output()
        # used to plot the mesh
        self.output_mesh = widgets.Output()

        self._build_widgets()

    def on_widget_changed(self, change):
        self.plot_geometry()

    def _build_widgets(self):

        offset_x = self.distance_x / 3.0
        offset_z = self.distance_x / 2.0

        self.widgets = {
            'but_gen_mesh': widgets.Button(
                description='Generate Mesh',
            ),
            'upper_left_x': widgets.FloatText(
                value=self.electrode_positions[0, 0] - offset_x,
                step=0.5,
                description='Upper left corner (X):',
            ),
            'upper_left_z': widgets.FloatText(
                value=self.electrode_positions[0, 1],
                step=0.5,
                description='Upper left corner (Z):',
            ),
            'lower_left_x': widgets.FloatText(
                value=self.electrode_positions[0, 0] - offset_x,
                step=0.5,
                description='Lower left corner (X):',
            ),
            'lower_left_z': widgets.FloatText(
                value=self.electrode_positions[:, 0].min() - offset_z,
                step=0.5,
                description='Lower left corner (Z):',
            ),
            'upper_right_x': widgets.FloatText(
                value=self.electrode_positions[-1, 0] + offset_x,
                step=0.5,
                description='Upper right corner (X):',
            ),
            'upper_right_z': widgets.FloatText(
                value=self.electrode_positions[-1, 1],
                step=0.5,
                description='Upper right corner (Z):',
            ),
            'lower_right_x': widgets.FloatText(
                value=self.electrode_positions[-1, 0] + offset_x,
                step=0.5,
                description='Lower right corner (X):',
            ),
            'lower_right_z': widgets.FloatText(
                value=self.electrode_positions[:, 0].min() - offset_z,
                step=0.5,
                description='Lower right corner (Z):',
            ),
            'char_length_1': widgets.FloatText(
                value=self.eldist_min / 3.0,
                step=0.1,
                description='Char length 1:',
            ),
        }
        self.widgets['but_gen_mesh'].on_click(self.generate_mesh)

        self.widgets['upper_left_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['upper_left_z'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_left_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_left_z'].observe(
            self.on_widget_changed, names='value')

        self.widgets['upper_right_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['upper_right_z'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_right_x'].observe(
            self.on_widget_changed, names='value')
        self.widgets['lower_right_z'].observe(
            self.on_widget_changed, names='value')

        gui = widgets.VBox([
            widgets.HBox([
                widgets.VBox(
                    [
                        widgets.HBox([
                            self.widgets['upper_left_x'],
                            self.widgets['upper_left_z'],
                        ]),
                        widgets.HBox([
                            self.widgets['lower_left_x'],
                            self.widgets['lower_left_z'],
                        ]),
                        widgets.HBox([
                            self.widgets['upper_right_x'],
                            self.widgets['upper_right_z'],
                        ]),
                        widgets.HBox([
                            self.widgets['lower_right_x'],
                            self.widgets['lower_right_z'],
                        ]),
                        self.widgets['char_length_1'],
                        self.widgets['but_gen_mesh'],
                    ]
                ),
                self.output_geometry,
            ]),
            self.output_mesh,
        ])
        self.gui = gui

    def show(self):
        self.output.clear_output()
        with self.output:
            display(self.gui)
        display(self.output)
        self.plot_geometry()

    def plot_geometry(self):
        with plt.ioff():
            fig, ax = plt.subplots()

            points_all = [
                [self.widgets[
                    'upper_left_x'].value, self.widgets['upper_left_z'].value],
            ]
            for electrode in self.electrode_positions:
                points_all += [electrode]

            points_all += [
                [self.widgets['upper_right_x'].value,
                 self.widgets['upper_right_z'].value],
                [self.widgets['lower_right_x'].value,
                 self.widgets['lower_right_z'].value],
                [self.widgets['lower_left_x'].value,
                 self.widgets['lower_left_z'].value],
            ]
            polygon = shapely.geometry.Polygon(points_all)
            self.polygon = polygon

            shapely.plotting.plot_polygon(polygon, ax=ax, linestyle='-')
            if polygon.is_simple:
                ax.set_title('Geometry looks valid!', loc='left')

            ax.scatter(
                self.electrode_positions[:, 0],
                self.electrode_positions[:, 1],
                label='electrodes',
                s=200,
                color='k',
            )
            # outer mesh corners
            ax.scatter(
                self.widgets['upper_left_x'].value,
                self.widgets['upper_left_z'].value,
                label='upper left',
                s=200,
            )
            ax.scatter(
                self.widgets['lower_left_x'].value,
                self.widgets['lower_left_z'].value,
                label='lower left',
                s=200,
            )
            ax.scatter(
                self.widgets['upper_right_x'].value,
                self.widgets['upper_right_z'].value,
                label='upper right',
                s=200,
            )
            ax.scatter(
                self.widgets['lower_right_x'].value,
                self.widgets['lower_right_z'].value,
                label='lower right',
            )

            ax.legend()
            ax.grid()
            self.output_geometry.clear_output()
            with self.output_geometry:
                display(fig)
            del (fig)

    def generate_mesh(self, button):
        print('Generating mesh')
        tempdir = tempfile.mkdtemp()
        # tempdir = 'tmp'
        print('PWD', os.getcwd())
        np.savetxt(
            tempdir + os.sep + 'electrodes.dat',
            self.electrode_positions, fmt='%.4f %.4f'
        )

        with open(tempdir + os.sep + 'boundaries.dat', 'w') as fid:
            fid.write('{} {} {}\n'.format(
                self.widgets['upper_left_x'].value,
                self.widgets['upper_left_z'].value,
                12
            ))
            for electrode in self.electrode_positions:
                fid.write('{} {} {}\n'.format(
                    *electrode, 12
                ))
            fid.write('{} {} {}\n'.format(
                self.widgets['upper_right_x'].value,
                self.widgets['upper_right_z'].value,
                11
            ))
            fid.write('{} {} {}\n'.format(
                self.widgets['lower_right_x'].value,
                self.widgets['lower_right_z'].value,
                11
            ))
            fid.write('{} {} {}\n'.format(
                self.widgets['lower_left_x'].value,
                self.widgets['lower_left_z'].value,
                11
            ))

        with open(tempdir + os.sep + 'char_length.dat', 'w') as fid:
            fid.write('{}\n'.format(
                self.widgets['char_length_1'].value
            ))

        pwd = os.getcwd()
        print(pwd)
        os.chdir(tempdir)
        try:
            subprocess.call('cr_trig_create grid', shell=True)
        except Exception:
            print('Caught an exception while calling cr_trig_create')
            os.chdir(pwd)
            return
        os.chdir('grid')
        self.mesh = crtomo.crt_grid()
        os.chdir(pwd)
        self.plot_mesh()

    def plot_mesh(self):
        if self.mesh is None:
            return
        with plt.ioff():
            fig, ax = plt.subplots()
            self.mesh.plot_grid_to_ax(ax)
        with self.output_mesh:
            self.output_mesh.clear_output()
            display(fig)


class geojson_from_zip_selector(object):
    def __init__(self, filename=None):
        self.widgets = {}
        self.filename = filename
        if filename is not None:
            print('loading .zip file')
            self.zfile = zipfile.ZipFile(filename)
        else:
            self.zfile = None

        # this holds the filename to the geojson file
        self.file_gj = None
        # data from the geojson file
        self.data_gj = None

        # we import wgs84 coordinates
        self.wgs84_points = []

        # utm coordinates are the coordinates we work with
        # note that all modifications are applied to the utm coords only
        # resetting is thus done by reconverting the wgs84 points to utm, if
        # required
        self.utm_coords = None
        self.utm_active_indices = None

        self.gui = None
        self._build_gui()

    def _zip_uploaded(self, change):
        if change['name'] == 'value':
            print(change['new'])
            cdict = change['new'][0]
            if cdict['type'] == 'application/zip' and cdict['size'] > 0:
                # try to load this as a zip
                self.zfile = zipfile.ZipFile(
                    io.BytesIO(cdict['content'])
                )
                # print('done loading .zip file')
                self._update_gui()

    def _selection_changed(self, change):
        if change['name'] == 'value':
            print('SELEcTION')
            print(change['new'])
            new_value = change['new']
            if new_value >= 0:
                self.widgets['but_load_gjfile'].disabled = False

    def _build_gui(self):
        self.widgets = {
            'upload': widgets.FileUpload(
                accept='.zip',
                multiple=False
            ),
            'gj_selector': widgets.RadioButtons(
                description='.geojson file to load:',
                disabled=False,
            ),
            'but_load_gjfile': widgets.Button(
                description='Load .geojson file',
                disabled=True,
            ),
            'output1': widgets.Output(),
            'output2': widgets.Output(),
            'output_utm_points': widgets.Output(),
            'but_show_el_manager': widgets.Button(
                description='Show Electrode Manager',
                disabled=True,
            ),
            'help_el_manager': widgets.HTML(
                value="<h2>Electrode manager</h2>" +
                "Arange electrode order",
                # placeholder='Some HTML',
                # description='Some HTML',
            ),
            'output_el_manager': widgets.Output(),
            'but_show_grid_creator': widgets.Button(
                description='Show Grid Creator',
                disabled=True,
            ),
            'help_grid_creator': widgets.HTML(
                value="<h2>Grid Creator</h2> " +
                "Create a CRTomo mesh",
                # placeholder='Some HTML',
                # description='Some HTML',
            ),
            'output_grid_creator': widgets.Output(),
        }

        self.gui = widgets.VBox([
            self.widgets['upload'],
            self.widgets['gj_selector'],
            self.widgets['but_load_gjfile'],
            GridBox(
                children=[
                    self.widgets['output1'],
                    self.widgets['output2'],
                ],
                layout=Layout(
                    width='100%',
                    grid_template_columns='auto 700px',
                    grid_template_rows='auto auto',
                    grid_gap='5px 10px'
                 )
            ),
            self.widgets['output_utm_points'],
            self.widgets['but_show_el_manager'],
            self.widgets['help_el_manager'],
            self.widgets['output_el_manager'],
            self.widgets['but_show_grid_creator'],
            self.widgets['help_grid_creator'],
            self.widgets['output_grid_creator'],
        ])

        self.widgets['upload'].observe(self._zip_uploaded)
        self.widgets['gj_selector'].observe(self._selection_changed)
        self.widgets['but_load_gjfile'].on_click(self.load_gjfile)

    def _prepare_utm_transformer(self):

        print('POINTS', self.wgs84_points)

        # First we need to get the UTM zone for our data points
        lon_min = self.wgs84_points[:, 0].min()
        lon_max = self.wgs84_points[:, 0].max()
        # xdiff = xmax - xmin

        lat_min = self.wgs84_points[:, 1].min()
        lat_max = self.wgs84_points[:, 1].max()

        print(lat_min, lat_max, lon_min, lon_max)

        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=lon_min,
                south_lat_degree=lat_min,
                east_lon_degree=lon_max,
                north_lat_degree=lat_max,
            ),
        )
        print('CODE', utm_crs_list[0].code)
        self.utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        print(self.utm_crs.to_wkt(pretty=True))

        # From SW Maps Homepage: All data recorded and exported by SW Maps is
        # in the WGS84 geographic
        # coordinate system (EPSG:4326).
        self.wgs84_crs = CRS.from_string("EPSG:4326")

        self.transformer = Transformer.from_crs(self.wgs84_crs, self.utm_crs)

    def _recompute_utm_distances(self):

        coords = self.utm_coords.copy()
        self.xy_distances_abs = np.hstack((
            0,
            np.sqrt(
                (coords[1:, 0] - coords[0:-1, 0]) ** 2 +
                (coords[1:, 1] - coords[0:-1, 1]) ** 2
            )
        ))

        for i in range(3):
            coords[:, i] -= coords[:, i].min()
        self.xy_distances_rel = np.hstack((
            0,
            np.sqrt(
                (coords[1:, 0] - coords[0:-1, 0]) ** 2 +
                (coords[1:, 1] - coords[0:-1, 1]) ** 2
            )
        ))
        print('Distances:')
        print(self.xy_distances_abs)
        print(self.xy_distances_rel)

    def _to_utm(self):
        if self.utm_coords is not None:
            # work was already done
            return
        self._prepare_utm_transformer()
        utm_coords = []
        for point in self.wgs84_points:
            utm_tmp = self.transformer.transform(point[1], point[0])
            utm_coords += [[utm_tmp[0], utm_tmp[1], point[2]], ]

        self.utm_coords = np.array(utm_coords)
        # by default, activate all electrodes
        self.utm_active_indices = list(range(self.utm_coords.shape[0]))
        self._recompute_utm_distances()

    def load_gjfile(self, button):
        # clean up any previous data
        self.wgs84_points = []
        self.utm_coords = None
        self.utm_active_indices = None

        # extract file into a buffer
        finfo = self.gjfiles[self.widgets['gj_selector'].value]

        buffer = io.BytesIO()
        buffer.write(self.zfile.read(finfo))
        buffer.seek(0)

        self.data_gj = geojson.load(buffer)

        self.wgs84_points = np.array(
            list(
                geojson.utils.coords(
                    self.data_gj['features']
                )
            )
        )
        self._to_utm()

        self.widgets['output1'].clear_output()
        self.widgets['output2'].clear_output()
        with self.widgets['output1']:
            with plt.ioff():
                fig, ax = self.plot_utm_to_map()
            display(fig)

        with self.widgets['output2']:
            with plt.ioff():
                fig, ax = self.plot_utm_topography(relative=True)
            display(fig)

        # show table with utm-datapoints
        self.point_widgets = []
        for index, point in enumerate(self.utm_coords):
            print('Creating label')
            label = widgets.Label(
                '{}: {}/{}/{} '.format(index, *point)
            )
            checkbox = widgets.Checkbox(
                value=True,
                description='Use as electrode',
                disabled=False,
                indent=False
            )
            button1 = widgets.Button(description='ignore all before')

            self.point_widgets += [label, checkbox, button1]
        gridbox = GridBox(
            children=self.point_widgets,
            layout=Layout(
                width='100%',
                grid_template_columns='250px 100px 100px',
                # grid_template_rows='80px auto 80px',
                # grid_gap='5px 10px'
            )
        )
        # vbox = widgets.VBox(self.point_widgets)
        self.widgets['output_utm_points'].clear_output()
        with self.widgets['output_utm_points']:
            display(gridbox)

        # now we can show the electrode manager
        self.widgets['but_show_el_manager'].disabled = False

        self.widgets['but_show_el_manager'].on_click(
            self._show_electrode_manager
        )

        self.widgets['but_show_grid_creator'].on_click(
            self._show_grid_creator
        )

    def _show_grid_creator(self, button):
        self.grid_creator = crtomo_mesh(
            self.el_manager.get_electrode_positions(),
            output=self.widgets['output_grid_creator'],
        )
        self.grid_creator.show()

    def _show_electrode_manager(self, button):
        electrode_positions = np.vstack((
            np.cumsum(self.xy_distances_rel),
            np.zeros(self.utm_coords.shape[0]),
            self.utm_coords[:, 2] - self.utm_coords[:, 2].min(),
        )).T
        self.el_manager = electrode_manager(
            electrode_positions,
            output=self.widgets['output_el_manager'],
        )
        self.el_manager.show()
        self.widgets['but_show_grid_creator'].disabled = False

    def _analyze_zfile(self):
        if self.zfile is None:
            return
        self.gjfiles = []
        for info in self.zfile.filelist:
            if info.file_size > 0 and info.filename.endswith('.geojson'):
                self.gjfiles += [info]

    def _update_gui(self):
        if self.zfile is None:
            # nothing to do
            return
        self._analyze_zfile()
        options = [(x.filename, index) for index, x in enumerate(self.gjfiles)]
        self.widgets['gj_selector'].options = options
        self.widgets['gj_selector'].value = None

    def show(self):
        self._update_gui()
        display(self.gui)

    def plot_utm_to_map(self):
        imagery = OSM(
            cache=True,
        )
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(
            1, 1, 1, projection=imagery.crs
        )

        # try to find some sane extents based on the loaded data points
        # longitude (east-west, x-axis)
        xmin = self.wgs84_points[:, 0].min()
        xmax = self.wgs84_points[:, 0].max()
        xdiff = xmax - xmin

        ymin = self.wgs84_points[:, 1].min()
        ymax = self.wgs84_points[:, 1].max()
        ydiff = ymax - ymin

        extent = [
            self.wgs84_points[:, 0].min() - xdiff / 2,
            self.wgs84_points[:, 0].max() + xdiff / 2,
            self.wgs84_points[:, 1].min() - ydiff / 2,
            self.wgs84_points[:, 1].max() + ydiff / 2,
        ]
        ax.set_extent(extent, ccrs.PlateCarree())
        # ax.set_extent(extent, ccrs.UTM(32))
        # [
        # Spiekeroog
        # 7.6587,
        # 7.84,
        # 53.75,
        # 53.79R

        # ], ccrs.PlateCarree()
        # )

        # Add the imagery to the map.
        detail = 19
        ax.add_image(imagery, detail)

        # skip the last character, which we implicitly assume to be N
        utm_zone = int(self.utm_crs.utm_zone[:-1])
        utm_projection = ccrs.UTM(utm_zone)

        for index, point in enumerate(self.utm_coords):
            ax.scatter(
                point[0],
                point[1],
                transform=utm_projection,
                s=100,
                color='r',
            )
            ax.scatter(
                point[1],
                point[0],
                transform=utm_projection,
                s=100,
                color='g',
            )

        for index, point in enumerate(self.wgs84_points):
            # ax.scatter(
            #     point[0],
            #     point[1],
            #     transform=ccrs.PlateCarree(),
            #     s=100,
            #     color='k',
            # )
            ax.annotate(
                '{}'.format(int(index)),
                xy=(point[0], point[1]),
                transform=ccrs.PlateCarree(),
                color='r',
                fontsize=26,
                bbox=dict(boxstyle="round", fc="0.8"),
                textcoords='offset pixels',
                xytext=(10, 25),
            )

        pstart = self.wgs84_points[0]
        pend = self.wgs84_points[-1]

        ax.plot(
            [pstart[0], pend[0]],
            [pstart[1], pend[1]],
            '-',
            transform=ccrs.PlateCarree(),
        )

        return fig, ax

    def plot_utm_topography(self, relative=False):
        coords = self.utm_coords.copy()
        if relative:
            for i in range(3):
                coords[:, i] -= coords[:, i].min()
            xy_distances = self.xy_distances_rel
        else:
            xy_distances = self.xy_distances_abs

        fig, axes = plt.subplots(1, 2)
        ax = axes[0]
        ax.plot(
            np.cumsum(xy_distances),
            coords[:, 2],
            '.-',
        )
        ax.grid()
        ax.set_xlabel('distance [m]')
        ax.set_ylabel('height [m]')
        if relative:
            ax.set_title('Relative heights', loc='left', fontsize=7)
        ax.set_title('Topography', loc='right')

        ax = axes[1]
        ax.plot(
            range(1, xy_distances.size),
            xy_distances[1:],
            '.-',
        )
        ax.grid()
        ax.set_ylabel('Distances to prev. el. [m]')
        ax.set_xlabel('Electrode Nr')
        return fig, ax


if __name__ == '__main__':
    gjsel = geojson_from_zip_selector('fake-ert GeoJSON.zip')
    gjsel.show()
