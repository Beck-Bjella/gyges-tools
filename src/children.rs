//! `gyges-tools children <in.csv> <out.csv>`: every legal next position of each input
//! position, for move-agreement tests.
//!
//! In:  `id,sq0..sq35` (side to move on the bottom, the data gen layout).
//! Out: `id,win,sq0..sq35` per legal move. The child is flipped to the opponent's view —
//! the same orientation as the next row of a data gen game — and `win` is 1 when the
//! move reaches the opponent's goal (the game ends there).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use gyges::{GenMoves, MoveGen, NoQuit, Player};

use crate::board::BoardState;

pub fn run(args: &[String]) {
    let usage = "usage: gyges-tools children <in.csv> <out.csv>";
    let (input, output) = (args.first().expect(usage), args.get(1).expect(usage));
    let reader = BufReader::new(File::open(input).expect("cannot open input"));
    let mut writer = BufWriter::new(File::create(output).expect("cannot create output"));
    let mut move_gen = MoveGen::default();
    let (mut positions, mut children) = (0usize, 0usize);

    for line in reader.lines() {
        let line = line.expect("read error");
        let mut fields = line.split(',');
        let id = fields.next().expect("missing id");
        let mut data = [0usize; 38];
        for (sq, v) in fields.take(36).enumerate() {
            data[sq] = v.trim().parse().expect("bad square value");
        }

        let mut board = gyges::BoardState::from(data);
        let mut move_list = unsafe { move_gen.gen::<GenMoves, NoQuit>(&mut board, Player::One) }.move_list;
        for mv in move_list.moves(&board) {
            let squares: Vec<usize> = mv.to_string().split('|').map(|s| s.parse().unwrap()).collect();
            let mut child = BoardState::from(data);
            child.make_move(&squares);
            child.flip();
            let row = child.data[..36].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            writeln!(writer, "{},{},{}", id, mv.is_win() as u8, row).unwrap();
            children += 1;
        }
        positions += 1;
    }
    println!("{} positions -> {} children ({:.1} per position)", positions, children,
             children as f64 / positions.max(1) as f64);
}
