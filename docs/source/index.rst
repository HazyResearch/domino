.. Domino documentation master file, created by
   sphinx-quickstart on Fri Jan  1 16:41:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to Domino
==========================================
Machine learning models that achieve high overall accuracy often make systematic errors 
on important slices of data. Domino provides toools to help discover these slices. 

Installation
~~~~~~~~~~~~

.. tabbed:: Main

    Domino is available on `PyPI <https://pypi.org/project/domino/>`_ and can be 
    installed with pip.

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

        pip install "domino @ git+https://github.com/HazyResearch/domino@dev"

    .. admonition:: Optional Dependencies
    
        Some parts of Domino rely on optional dependencies. To install all optional
        dependencies use: 

        .. code-block:: bash

            pip install "domino[all] @ git+https://github.com/HazyResearch/domino@dev"
        
        You can also install specific groups optional of dependencies using something like: 

        .. code-block:: bash

            pip install "domino[text] @ git+https://github.com/HazyResearch/domino@dev"
            
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



Next Steps
~~~~~~~~~~~~

.. panels::

    Get started with Domino by following along on Google Colab. 

    .. link-button:: https://drive.google.com/file/d/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM/view?usp=sharing 
        :classes: btn-primary btn-block stretched-link
        :text: Walkthrough Notebook
    ---

    Learn more about the motivation behind Domino and what it enables. 

    .. link-button:: https://www.notion.so/sabrieyuboglu/Domino-DataPanels-for-Machine-Learning-64891aca2c584f1889eb0129bb747863
        :classes: btn-primary btn-block stretched-link
        :text: Introductory Blog Post 


.. _Issues: https://github.com/HazyResearch/domino/issues/


.. toctree::
   :hidden:
   :maxdepth: 2

   intro.md


.. toctree::
    :hidden:
    :maxdepth: 2

    apidocs/index


..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`







