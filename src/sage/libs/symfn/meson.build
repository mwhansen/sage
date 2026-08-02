py.install_sources(
  '__init__.py',
  'backend.py',
  'extras.py',
  'terms.pyx',
  subdir: 'sage/libs/symfn',
)

extension_data = {'terms': files('terms.pyx')}

foreach name, pyx : extension_data
  py.extension_module(
    name,
    sources: pyx,
    subdir: 'sage/libs/symfn',
    install: true,
    include_directories: [inc_cpython, inc_rings],
    dependencies: [py_dep, gmp],
  )
endforeach
