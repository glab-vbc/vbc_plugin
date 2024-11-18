# Plugin for Custom Square Density Potential for Hoomd

To install a plugin in hoomd you first have to build hoomd from source, because when installing the plugin it uses the cmake files from hoomd installation. Here are the instructions to build hoomd from [source](https://github.com/glotzerlab/hoomd-blue/blob/trunk-patch/BUILDING.rst). Don't forget to put `-DENABLE_GPU=on` flag while building with cmake, if you want to install it with GPU support.

>[!Note]
>Sometimes you might get some problem with eigen package when building, that cmake can't fine some files for Eigen3, if that happens try installing eigen through conda-forge channel (`conda install -c conda-forge eigen`), as it will also install all the required cmake files for Eigen3 package in appropriate directories.

Now, after successfully building the hoomd from source, follow the same instructions for installing the plugin.

1. Clone the repository
   ```bash
   git clone https://github.com/glab-vbc/sq_density_plugin.git
   ```
2. Change directory
   ```bash
   cd sq_density_plugin
   ```
3. Configure
   ```bash
   mkdir build
   cd build
   cmake ..
   ```
4. Build
   ```bash
   make
   ```
5. Install
   ```bash
   make install
   ```

After following the above steps you can check if the installation worked properly by importing the module

```python
import hoomd
from hoomd.sqd_plugin import sqd_potential

nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=('bond',))
sqd = sqd_potential.SquareDensity(nlist, default_r_cut=3.0)
```
