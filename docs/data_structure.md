- RootReader
  - file_readers
    - reader
      - groups
        - group
          - coord: (nframe, natom, 3)
          - cell: (nframe, 3, 3)
          - atom_ids: (natom,)
          - chg: (nframe, nx, ny, nz)
          - chgdiff: (nframe, nx, ny, nz)
          - sample_grids: (ngrid, 3)
          - sample_chg: (ngrid,)
          - sample_chgdiff: (ngrid,)
    
        