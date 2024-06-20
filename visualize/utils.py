import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from ase.io import read, write
from ase import Atoms
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from pymatgen.core import Structure, Molecule


def data2atoms(data, gmae=False, correct=True):
    if gmae:
        atoms = Atoms(
            positions=data.gmae_pos.tolist(),
            numbers=data.gmae_atom_numb.long().tolist(),
            cell=data.cell[0].tolist(),
            pbc=(1, 1, 1),
        )
        return atoms
    else:
        atoms = Atoms(
            positions=data.pos.tolist(),
            numbers=data.atomic_numbers.long().tolist(),
            cell=data.cell[0].tolist(),
            pbc=(1, 1, 1),
        )

    if 'cross_weights' in data.keys:
        if correct and 'tags' in data.keys:
            weights = data.cross_weights
            weights[data.tags == 0] = 0.0
            weights = weights / weights.sum()
            atoms.set_array('cross_weights', weights.tolist(), float)
        else:
            atoms.set_array('cross_weights', data.cross_weights.tolist(), float)

    return atoms


def data2struc(data):
    struc = Structure(
        lattice=data.cell.tolist(),
        species=data.atomic_numbers.long().tolist(),
        coords=data.pos.tolist(),
        coords_are_cartesian=True,
    )
    return struc


def data2poscar(data, oname='data.vasp'):
    atoms = data2atoms(data)
    write(oname, atoms, format='vasp', sort=True, vasp5=True)
    return 0


def data2cif(data, oname='data.icf'):
    atoms = data2atoms(data)
    write(oname, atoms, format='cif')
    return 0


def map_view(atoms, prop, xsize=400, ysize=300, unitcell=True,
             cm="plasma", minval=None, maxval=None, label=None):
    """ Map the given properties on a color scale on to the structure.
    The properties can be either a prop name, or an array of values.
    In the last case, the array size must be consistent with atom number.

    Args:
        atoms (ase.Atoms): input atoms for nglview
        prop (str or array): name of the properties or values you want to map
        xsize (int): width of the nglview widget, default '400'
        ysize (int): height of the nglview widget, default '300'
        unitcell (bool): If True and structure is periodic, show the unitcell.
        cm (str): colormap from ``matplotlib.cm``.
        minval (float): minimum value to consider for the color sacle
        maxval (float): maximum value to consider for the color sacle
        label (str): Name of the colorbar. If None, use prop.

    Returns:
        Returns an ipywidgets ``HBox`` or ``VBox`` with the ``NGLWidget``
        and a color bar associated to the mapped properties. The 
        ``NGLWidget`` is the first element of the children, the colorbar
        is the second one.
    """

    # try to import nglview and ipywidgets
    try:
        import nglview
        from ipywidgets import HBox, VBox, Output
    except ImportError as e:
        print("You need nglview and ipywidgets available with jupyter notebook.")
        print(e)
        return None

    # check property data
    if isinstance(prop, str):
        try:
            prop_vals = atoms.get_array(prop)
            label = prop if label is None else label
        except ValueError:
            raise ValueError("prop %s not found in Atoms." % prop)
    else:
        try:
            prop_vals = np.array(prop, dtype=np.float64).reshape(len(atoms))
        except ValueError:
            print("property = ", prop)
            raise ValueError(
                "Cannot convert prop in a numpy array of floats.")

        # colorbar label
        label = "" if label is None else label

    # find property boundary
    if minval is None:
        minval = np.nanmin(prop_vals)
    if maxval is None:
        maxval = np.nanmax(prop_vals)
    assert maxval > minval

    # normalize colors
    normalize = mpl.colors.Normalize(minval, maxval)
    cmap = mpl.cm.get_cmap(cm)
    # set up a matplotlib figure for the colorbar (vertical)
    _, ax = plt.subplots(figsize=(0.3, 3))
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=normalize,
                              orientation="vertical")
    ax.set_title(label)

    # place the colorbar in an Output() widget
    cb = Output()
    with cb:
        plt.show()

    # get NGL View from ase.Atoms
    view = nglview.show_ase(atoms, default=False)
    view.clear_representations()

    # set the atom colors
    for iat, val in enumerate(prop_vals):
        if np.isnan(val):
            continue
        color = mpl.colors.rgb2hex(cmap(X=normalize(val), alpha=1))
        view.add_spacefill(selection=[iat],
                           radiusType='covalent',
                           radiusScale=1.0,
                           color=color,
                           color_scale='rainbow')

    # view setting
    if unitcell:
        view.add_unitcell()
    view.camera = 'orthographic'
    view.parameters = {"clipDist": 0}
    view.center()
    # resize nglview widget
    view._remote_call('setSize', target='Widget',
                      args=['%dpx' % (xsize,), '%dpx' % (ysize,)])

    # gather the view and colorbar in a hbox
    gui = HBox([view, cb])
    # Make useful shortcuts for the user of the class
    gui.view = view

    return gui


def atoms_view(atoms, xsize=400, ysize=300, unitcell=True):
    """
    Args:
        atoms (ase.Atoms): input atoms for nglview
        xsize (int): width of the nglview widget, default '400'
        ysize (int): height of the nglview widget, default '300'
        unitcell (bool): If True and structure is periodic, show the unitcell.

    Returns:
        NGLWidget
    """

    # try to import nglview and ipywidgets
    try:
        import nglview
        from ipywidgets import HBox
    except ImportError as e:
        print("You need nglview and ipywidgets available with jupyter notebook.")
        print(e)
        return None

    # get NGL View from ase.Atoms
    view = nglview.show_ase(atoms, default=False)
    view.clear_representations()

    # atom setting
    view.add_spacefill(radiusType='covalent',
                       radiusScale=1.0,
                       color_scheme='element',
                       color_scale='rainbow')

    # view setting
    if unitcell:
        view.add_unitcell()
    view.camera = 'orthographic'
    view.parameters = {"clipDist": 0}
    view.center()
    # resize nglview widget
    view._remote_call('setSize', target='Widget',
                      args=['%dpx' % (xsize,), '%dpx' % (ysize,)])

    # gather the view and colorbar in a hbox
    gui = HBox([view])
    # Make useful shortcuts for the user of the class
    gui.view = view

    return gui


def get_dft_data(targets):
    """
    Get DFT data of original OC20Dense dataset
    Organizes the released target mapping for evaluation lookup.

    oc20dense_targets.pkl:
        ['system_id 1': [('config_id 1', dft_adsorption_energy), ('config_id 2', dft_adsorption_energy)], `system_id 2]

    Returns: Dict:
        {
           'system_id 1': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
           'system_id 2': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
           ...
        }
    """
    dft_data = defaultdict(dict)
    for system in targets:
        for adslab in targets[system]:
            dft_data[system][adslab[0]] = adslab[1]

    return dft_data
