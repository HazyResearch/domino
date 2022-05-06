Getting started
----------------


Installation
~~~~~~~~~~~~

.. tabbed:: Main

    Domino is available on `PyPI <https://pypi.org/project/domino/>`_ and can be 
    installed with pip.

    .. warning::

        Domino is currently being actively developed. We recommend installing using
        the "Latest" or "Editable" tabs for the most up to date version.

    .. code-block:: 

        pip install domino
    
    .. admonition:: Optional dependencies
    
        Some parts of Domino rely on optional dependencies. To install all optional
        dependencies use: 

        .. code-block:: bash
        
            pip install domino[all] 
            
        You can also install specific
        groups optional of dependencies using something like: 

        .. code-block:: bash
        
            pip install domino[text]
        
        See `setup.py` for a full list of 
        optional dependencies.   

.. tabbed:: Latest
    
    To install the latest development version of Domino use:

    .. code-block:: bash

        pip install "domino @ git+https://github.com/HazyResearch/domino@main"

    .. admonition:: Optional Dependencies
    
        Some parts of Domino rely on optional dependencies. To install all optional
        dependencies use: 

        .. code-block:: bash

            pip install "domino[all] @ git+https://github.com/HazyResearch/domino@main"
        
        You can also install specific groups optional of dependencies using something like: 

        .. code-block:: bash

            pip install "domino[text] @ git+https://github.com/HazyResearch/domino@main"
            
        See `setup.py` for a full list of optional dependencies.   

.. tabbed:: Editabled

    To install from editable source, clone the domino repository and pip install in
    editable mode. 

    .. code-block:: bash

        git clone https://github.com/HazyResearch/domino.git
        cd domino
        pip install -e .

    .. admonition:: Optional Dependencies
    
        Some parts of Domino rely on optional dependencies. To install all optional
        dependencies use: 

        .. code-block:: bash

            pip install -e .[dev]
        
        You can also install specific groups optional of dependencies using something like: 

        .. code-block:: bash

            pip install -e .[text]
            
        See `setup.py` for a full list of optional dependencies.   


