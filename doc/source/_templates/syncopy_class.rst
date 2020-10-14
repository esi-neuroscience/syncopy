{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :private-members:
   
   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods|unique %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   .. autosummary::
   {% for item in members %}
      {% if item[0] == "_" %}
         {% if item[1] != "_" %}
            ~{{ name }}.{{ item }}
         {% endif %}
      {% endif %}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
