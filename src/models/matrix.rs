#[derive(Clone)]
pub struct Matrix {
    pub fields: Vec<f64>,
    pub rows: usize,
    pub columns: usize
}

impl Matrix {
    pub fn full_of_zero(rows: usize, columns: usize) -> Matrix {

        Matrix {
            fields: vec![0.0; rows * columns],
            rows: rows,
            columns: columns
        }         
    }

    pub fn full_of_rands(rows: usize, column: usize) -> Matrix  {

        let mut vec: Vec<f64> = vec![];

        for _ in 0..rows * column {
            vec.push(rand::random());
        }

        Matrix {
            fields: vec,
            rows: rows,
            columns: column
        }
    }

    pub fn dot_product(&self, other: &Matrix) -> Matrix {
        if self.columns != other.rows {
            panic!("Column of the first and rows of the second MUST be equal");
        }

        let mut matrix_full_of_zero = Matrix::full_of_zero(self.rows, other.columns);

        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut sum = 0.0;

                for k in 0..self.columns {
                    sum += self.fields[i * self.columns + k] * other.fields[k * other.columns + j];
                }

                matrix_full_of_zero.fields[i * other.columns + j] = sum;
            }
        }
        matrix_full_of_zero
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Rows and Columns MUST be equal");
        }

        let mut matrix = Matrix::full_of_zero(self.rows, self.columns);

        for k in 0..self.fields.len() {
            matrix.fields[k] = self.fields[k] + other.fields[k];
        }

        matrix
    }

    pub fn apply<F>(&self, func: F) -> Matrix 
    where 
        F: Fn(f64) -> f64
    {

        let mut matrix = Matrix::full_of_zero(self.rows, self.columns);

        for k in 0..self.fields.len() {
            matrix.fields[k] = func(self.fields[k]);
        }

        matrix
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Rows and Columns MUST be equal");
        }
        let mut matrix = Matrix::full_of_zero(self.rows, self.columns);

        for k in 0..self.fields.len() {
            matrix.fields[k] = self.fields[k] - other.fields[k];
        }

        matrix
    }


    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.columns != other.columns {
            panic!("Rows and Columns MUST be equal");
        }
        let mut matrix = Matrix::full_of_zero(self.rows, self.columns);

        for k in 0..self.fields.len() {
            matrix.fields[k] = self.fields[k] * other.fields[k];
        }

        matrix
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::full_of_zero(self.columns, self.rows);

        for i in 0..self.rows {
           for j in 0..self.columns {
                let origin_idx = i * self.columns + j;
                let value = self.fields[origin_idx];

                let new_idx = j * self.rows + i;

                result.fields[new_idx] = value
           } 
        }

        result
    }
}
