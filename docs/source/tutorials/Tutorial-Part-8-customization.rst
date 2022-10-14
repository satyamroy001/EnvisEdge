Customization
=============

Training Configuration
----------------------

There are two ways to adjust training hyper-parameters: - \**Set values
in config/*.yml\ **persistent settings which are necessary for
reproducibility eg randomization seed -**\ Pass them as CLI argument*\*
Good for non-persistent and dynamic settings like GPU device

*In case of conflict, the CLI argument supersedes the config file parameter.*
For further reference, check out `training config
flags <configs/flags.md>`__

Model Architecture
------------------

Adjusting DLRM model params
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any parameter needed to instantiate the PyTorch module can be supplied
by simply creating a key-value pair in the config file.

For example DLRM requires ``arch_feature_emb_size``, ``arch_mlp_bot``,
etc

.. code:: 

   model: 
     name : 'dlrm'
     arch_sparse_feature_size : 16
     arch_mlp_bot : [13, 512, 256, 64]
     arch_mlp_top : [367, 256, 1]
     arch_interaction_op : "dot"
     arch_interaction_itself : False
     sigmoid_bot : "relu"
     sigmoid_top : "sigmoid"
     loss_function: "mse"

Adding new models
~~~~~~~~~~~~~~~~~

Model architecture can only be changed via ``configs/*.yml`` files.
Every model declaration is tagged with an appropriate name and loaded
into the registry.

.. code:: python

   @registry.load('model','<model_name>')
   class My_Model(torch.nn.Module):
       def __init__(num):
           ... 

You can define your own modules and add them in the
`fedrec/modules <fedrec/modules>`__. Finally set the ``name`` flag of
``model`` tag in the config file

.. code::

   model : 
     name : "<model name>"
