"""Tantrix solver."""

import argparse
import numpy as np
from scipy.ndimage import morphology
import svgwrite
import time
from typing import List, Optional, Sequence, Tuple


class Piece(object):
  """Single Tantrix piece.

  Attrs:
    name: Arbitrary nickname.
    colors: String of length 6 indicating the color at each edge.
      e.g., piece #1 is 'YYBRBR'.
    rotation: Current rotation, an int in range 0-5.
      A value of N indicates that colors should be shifted counterclockwise
      by N, e.g. rotation of 1 converts 'ABCDEF' to 'FABCDE'.
    coord: Tuple (row,col) indicating board position, or None if not placed
      on board.
  """

  def __init__(self, name: str, colors: str):
    self.name = name
    self.colors = colors
    self.rotation = 0
    self.coord = None

  def __str__(self):
    return '%s [%s]' % (
        self.name, (self.colors[self.rotation:] + self.colors[:self.rotation]))

  def get_color(self, direction: int):
    return self.colors[(self.rotation + direction) % 6]


pieces = [
    Piece('#1', 'YYBRBR'),
    Piece('#2', 'YYBRRB'),
    Piece('#3', 'YYRRBB'),
    Piece('#4', 'YBYRBR'),
    Piece('#5', 'YYRBBR'),
    Piece('#6', 'YBRYRB'),
    Piece('#7', 'YRBBYR'),
    Piece('#8', 'YBBRYR'),
    Piece('#9', 'YRYBRB'),
    Piece('#10', 'YYRBRB'),
    Piece('#11', 'YBYRRB'),
    Piece('#12', 'YBYBRR'),
    Piece('#13', 'YBBYRR'),
    Piece('#14', 'YYBBRR'),
    Piece('#15', 'RYYRGG'),
    Piece('#16', 'YGGYRR'),
    Piece('#17', 'YYGRGR'),
    Piece('#18', 'YYRGRG'),
    Piece('#19', 'RRGYGY'),
    Piece('#20', 'YGYGRR'),
]
cycle_colors = [
    None,
    None,
    'Y',
    'R',
    'R',
    'B',
    'B',
    'B',
    'Y',
    'Y',
    'R',
    'Y',
    'B',
    'B',
    'R',
    'R',
    'Y',
    'R',
    'R',
    'Y',
]
neighbor_delta = np.array([[-1, +1],   # A
                           [ 0, +1],   # B
                           [+1,  0],   # C
                           [+1, -1],   # D
                           [ 0, -1],   # E
                           [-1,  0]])  # F


class Board(object):
  """Board of Tantrix pieces.

  Attrs:
    grid: Array of currently placed Piece objects.
  """

  def __init__(self):
    self.grid = np.zeros((30, 30), dtype=np.object)

  def is_valid_placement(self, coord: Tuple[int, int], piece: Piece) -> bool:
    """Determines whether `piece` can be placed at `coord` with some rotation.

    Returns:
      If piece can be placed at the given coordinates, rotates the piece
      accordingly and returns True. Otherwise, returns False.
    """
    if self.grid[coord]:
      return False  # Coord is already taken.
    for rot in range(6):
      piece.rotation = rot
      if self.is_valid_placement_and_rotation(coord, piece):
        print('Piece can be placed with rotation %d' % rot)
        return True
    return False

  def is_valid_placement_and_rotation(self, coord: Tuple[int, int],
                                      piece: Piece) -> bool:
    """Determines whether `piece` can be placed at `coord` with given rotation.
    """
    if self.grid[coord]:
      return False  # Coord is already taken.

    # Check validity at every edge.
    for direction in range(6):
      neighbor_coord = coord + neighbor_delta[direction, :]
      if (np.any(neighbor_coord < 0) or
          np.any(neighbor_coord >= self.grid.shape)):
        # Neighbor is out of bounds, so no objections in this direction.
        continue
      neighbor_coord = tuple(neighbor_coord)
      if not self.grid[neighbor_coord]:
        # Neighbor is unoccupied, so no objections in this direction.
        continue
      my_color = piece.get_color(direction)
      neighbor_color = self.grid[neighbor_coord].get_color(direction + 3)
      if my_color != neighbor_color:
        # print('Direction %d: My color %s does not match neighbor %s color %s'
        #       % (direction, my_color, self.grid[neighbor_coord],
        #          neighbor_color))
        return False
      # else:
      #   print('Direction %d: My color %s matches neighbor %s' % (
      #       direction, my_color, self.grid[neighbor_coord]))

    return True

  def place_piece(self, coord: Tuple[int, int], piece: Piece) -> None:
    self.grid[coord] = piece
    piece.coord = coord

  def remove_piece(self, coord: Tuple[int, int]) -> None:
    self.grid[coord].coord = None
    self.grid[coord] = 0

  def has_holes(self) -> bool:
    piece_locations = (self.grid != 0)
    filled_locations = morphology.binary_fill_holes(piece_locations)
    return np.any(piece_locations != filled_locations)

  def compress(self) -> None:
    """Shrink grid to smallest possible size."""
    for i in range(self.grid.shape[1]):
      if not np.any(self.grid[:, i:]):
        self.grid = self.grid[:, :i]
        break
    for i in range(self.grid.shape[1], 0, -1):
      if not np.any(self.grid[:, :i]):
        self.grid = self.grid[:, i:]
        break
    for i in range(self.grid.shape[0]):
      if not np.any(self.grid[i:, :]):
        self.grid = self.grid[:i, :]
        break
    for i in range(self.grid.shape[0], 0, -1):
      if not np.any(self.grid[:i, :]):
        self.grid = self.grid[i:, :]
        break

  def to_svg(self) -> str:
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    hex_height = 30
    hex_width = hex_height * np.sin(np.pi/3)
    hex_coords = np.asarray([np.sin(angles), np.cos(angles)]).T * hex_height / 2
    hex_coords += [hex_width/2, hex_height/2]
    hex_mid_coords = np.mean([hex_coords, np.roll(hex_coords, 1, axis=0)], axis=0)
    hex_mid_coords = hex_mid_coords[[3,2,1,0,5,4],:]
    hex_shift_mat = np.asarray([[hex_width, hex_width/2], [0, hex_coords[1,1]]])

    color_table = {
        'R': '#f00',
        'B': '#00f',
        'Y': '#ff0',
        'G': '#0f0'
    }

    page_dims = hex_shift_mat @ self.grid.shape[::-1] + [hex_width, hex_height]
    page_width, page_height = page_dims
    dwg = svgwrite.Drawing(size=('%dmm' % page_width, '%dmm' % page_height),
                           viewBox='0 0 %d %d' % (page_width, page_height))
    for y, row in enumerate(self.grid):
      for x, cell in enumerate(row):
        if not cell:
          continue
        grp = svgwrite.container.Group()
        grp.add(dwg.polygon(hex_coords, stroke='none', fill='#000'))

        # Colors
        completed_dirs = set()
        dir_to_color = [cell.get_color(i) for i in range(6)]
        for incoming_dir in range(6):
          if incoming_dir in completed_dirs: continue
          color = dir_to_color[incoming_dir]
          outgoing_dir = dir_to_color.index(color, incoming_dir+1)
          hex_center = np.mean(hex_coords, axis=0)
          start = hex_mid_coords[incoming_dir, :]
          end = hex_mid_coords[outgoing_dir, :]
          control1 = np.mean([start, hex_center], axis=0)
          control2 = np.mean([end, hex_center], axis=0)
          grp.add(dwg.path(
              d='M%f,%f C%f,%f %f,%f %f,%f' % (
                  start[0], start[1], control1[0], control1[1],
                  control2[0], control2[1], end[0], end[1]),
              stroke=color_table[color],
              stroke_width=1.5))
          completed_dirs.add(incoming_dir)
          completed_dirs.add(outgoing_dir)

        grp.add(dwg.polygon(hex_coords,
                            stroke='#fff', stroke_width='0.9', fill='none'))
        grp.add(dwg.text('%s'%cell.name, hex_center, font_size='3', fill='#fff'))

        transform = np.matmul(hex_shift_mat, [[x],[y]])
        grp.translate(tx=transform[0], ty=transform[1])
        dwg.add(grp)

    return dwg.tostring()

def get_outgoing_dir(piece: Piece, incoming_dir: int) -> int:
  color = piece.get_color(incoming_dir)
  direction = (piece.colors.index(color) - piece.rotation) % 6
  if direction == incoming_dir:
    direction = (piece.colors.index(color, (direction + piece.rotation) % 6 + 1) - piece.rotation) % 6
  return direction


def find_cycle(pieces: List[Piece], cycle_color: str, allow_holes: bool
              ) -> Optional[Board]:
  """Solve a Tantrix puzzle.

  Args:
    pieces: List of pieces to use. Their locations and rotations will be
      updated.
    cycle_color: Color of requested cycle.
    allow_holes: Whether holes are allowed in the solution.

  Returns:
    Shallow copy of list of pieces, along with their placement in

  Raises:
    RuntimeError: If puzzle is unsolveable.
  """
  board = Board()
  start_time = time.time()
  start_coord = (15, 15)
  n_attempts = [0]

  def recurse(depth: int, coord: Tuple[int,int], piece_idx: int, incoming_dir: int) -> bool:
    board.place_piece(coord, pieces[piece_idx])
    n_attempts[0] += 1
    if not allow_holes and board.has_holes():
        board.remove_piece(coord)
        return False  # Placed all pieces, but did not form a cycle.

    # print('%sPlaced %s' % ('  ' * depth, pieces[piece_idx]))
    outgoing_dir = get_outgoing_dir(pieces[piece_idx], incoming_dir)
    next_coord = tuple(coord + neighbor_delta[outgoing_dir, :])

    if piece_idx == len(pieces) - 1:
      # Last piece. Check if a cycle is formed.
      if next_coord == start_coord:
        # print('%sSolved!' % ('  ' * depth))
        return True  # Solved!
      else:
        # print('%sNot a cycle' % ('  ' * depth))
        # print('%sRemoved %s' % ('  ' * depth, pieces[piece_idx]))
        board.remove_piece(coord)
        return False  # Placed all pieces, but did not form a cycle.

    for next_piece_idx in range(piece_idx + 1, len(pieces)):
      pieces[piece_idx + 1], pieces[next_piece_idx] = pieces[next_piece_idx], pieces[piece_idx + 1]
      next_piece = pieces[piece_idx + 1]
      for rotation in range(6):
        next_piece.rotation = rotation
        if board.is_valid_placement_and_rotation(next_coord, next_piece):
          if recurse(depth + 1, next_coord, piece_idx + 1, (outgoing_dir + 3) % 6):
            return True  # Solved
      pieces[piece_idx + 1], pieces[next_piece_idx] = pieces[next_piece_idx], pieces[piece_idx + 1]

    # Dead end. Undo changes to board.
    # print('%sRemoved %s' % ('  ' * depth, pieces[piece_idx]))
    board.remove_piece(coord)
    return False

  if recurse(0, start_coord, 0, pieces[0].colors.index(cycle_color)):
    end_time = time.time()
    print('== Solved in %d attempts (%.1f seconds) ==' % (
        n_attempts[0], end_time - start_time))
    for piece in pieces:
      print('%s @%s' % (piece, piece.coord))
    board.compress()
    return board

  print('Failed to solve.')
  return None


def main():
  parser = argparse.ArgumentParser(description='Tantrix puzzle solver.')
  parser.add_argument(
      '--num_pieces', type=int, default=10,
      help='Number of pieces in the puzzle.')
  parser.add_argument(
      '--allow_holes', type=bool, default=True,
      help='Whether to allow holes in the solution. Disallowing holes increases solution time.')
  parser.add_argument(
      '--output_svg', type=str, default=None, help='Output SVG file containing solution.')
  args = parser.parse_args()

  board = find_cycle(
      pieces[:args.num_pieces],
      cycle_colors[args.num_pieces - 1],
      allow_holes=args.allow_holes)
  if args.output_svg:
    with open(args.output_svg, 'wt') as f:
      f.write(board.to_svg())


if __name__ == '__main__':
  main()

