use std::fmt::Display;

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
    
    pub fn make_move(&mut self, mv: Vec<usize>) {
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
            let step3_piece = self.data[step3];

            self.data[step1] = 0;
            self.data[step2] = step1_piece;
            self.data[step3] = step2_piece;
            

        } 


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