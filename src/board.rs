use std::fmt::Display;

use gyges::{GenMoves, MoveGen, NoQuit, Player};

/// Board in the engine's `setpos data` layout: squares 0..36 row by row from the
/// side to move's home row, 36 = own goal, 37 = opponent's goal. The side to
/// move is always on the bottom; `flip` hands the board to the other player.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoardState {
    pub data: [usize; 38],

}

impl BoardState {
    pub fn new() -> BoardState {
        BoardState {
            data: [0; 38],

        }

    }

    pub fn make_move(&mut self, mv: &[usize]) {
        if mv.len() == 2 {
            let step1 = mv[0];
            let step2 = mv[1];

            let step1_piece = self.data[step1];

            self.data[step1] = 0;
            self.data[step2] = step1_piece;

        } else if mv.len() == 3 {
            let step1 = mv[0];
            let step2 = mv[1];
            let step3 = mv[2];

            let step1_piece = self.data[step1];
            let step2_piece = self.data[step2];

            self.data[step1] = 0;
            self.data[step2] = step1_piece;
            self.data[step3] = step2_piece;

        }

    }

    /// True if `mv` is a legal move for the side to move, according to the
    /// `gyges` library's move generator. Engines are never trusted to follow
    /// the rules; this is what keeps the simulation on the current rule set
    /// (e.g. the 2.0.0 rule that every move must change the board).
    pub fn is_legal(&self, mv: &[usize]) -> bool {
        let mv_str = mv.iter().map(|s| s.to_string()).collect::<Vec<_>>().join("|");

        let mut board = gyges::BoardState::from(self.data);
        let mut move_gen = MoveGen::default();
        let mut move_list = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, Player::One) }.move_list;

        move_list.moves(&board).iter().any(|m| m.to_string() == mv_str)

    }

    pub fn flip(&mut self) {
        let mut temp_data: [usize; 38] = [0; 38];

        for y in 0..6 {
            for x in 0..6 {
                let piece = self.data[y * 6 + x];

                temp_data[((5 - y) * 6) + (5 - x)] = piece;

            }

        }

        self.data = temp_data;

    }

}

impl From<[usize; 38]> for BoardState {
    fn from(data: [usize; 38]) -> Self {
        BoardState {
            data,

        }

    }

}

impl Display for BoardState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.data[37] == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.data[37])?;

        }
        writeln!(f, " ")?;
        writeln!(f, " ")?;

        for y in (0..6).rev() {
            for x in 0..6 {
                if self.data[y * 6 + x] == 0 {
                    write!(f, "    .")?;
                } else {
                    write!(f, "    {}", self.data[y * 6 + x])?;

                }

            }
            writeln!(f, " ")?;
            writeln!(f, " ")?;

        }

        writeln!(f, " ")?;
        if self.data[36] == 0 {
            writeln!(f, "                .")?;

        } else {
            writeln!(f, "                {}", self.data[36])?;

        }

        Result::Ok(())

    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn start() -> BoardState {
        let mut b = BoardState::new();
        b.data[..6].copy_from_slice(&[3, 2, 1, 1, 2, 3]);
        b.data[30..36].copy_from_slice(&[3, 2, 1, 1, 2, 3]);
        b
    }

    #[test]
    fn accepts_a_normal_move() {
        // The 3 on square 0 can step straight up three squares.
        assert!(start().is_legal(&[0, 18]));
    }

    #[test]
    fn rejects_unreachable_and_garbage_moves() {
        assert!(!start().is_legal(&[0, 35]));
        assert!(!start().is_legal(&[100, 100, 100]));
        assert!(!start().is_legal(&[]));
    }

    #[test]
    fn rejects_moves_that_leave_the_board_unchanged() {
        // 1s on squares 0 and 1. The 1 on 0 lands on the 1 on 1, picks it up and
        // drops it back on 0: same piece types, so the board is unchanged.
        let mut b = BoardState::new();
        b.data[..6].copy_from_slice(&[1, 1, 2, 2, 3, 3]);
        b.data[30..36].copy_from_slice(&[3, 2, 1, 1, 2, 3]);
        assert!(!b.is_legal(&[0, 1, 0]));
        // Or bounces off it straight back onto its own start square.
        assert!(!b.is_legal(&[0, 0]));
        assert!(b.is_legal(&[0, 6]));
    }
}
