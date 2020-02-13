package pl.marek;

class GreedyAlgorithm {

    private Integer[][][] grid;

    GreedyAlgorithm(Integer[][][] grid) {
        this.grid = grid;
    }

    int solve(int endX, int endY) {
        int currentX = 0;
        int currentY = 0;

        System.out.println("Greedy - Calculate path to point : " + "[" + endX + ", " + endY + "]" );
        System.out.println("[" + currentX + ", " + currentY + "]");

        int cost = 0;

        while (!checkPointIsReached(currentX, endX, currentY, endY)) {
            if (currentX >= endX) {
                cost += grid[currentY][currentX][0];
                currentY += 1;
            } else if (currentY >= endY) {
                cost += grid[currentY][currentX][1];
                currentX += 1;
            } else {
                int costToUp = grid[currentY][currentX][0];
                int costToRight = grid[currentY][currentX][1];

                if (costToUp >= costToRight) {
                    cost += costToRight;
                    currentX += 1;
                } else {
                    cost += costToUp;
                    currentY += 1;
                }
            }
            System.out.println("[" + currentX + ", " + currentY + "]");
        }
        System.out.println("Cost : " + cost);
        System.out.println();
        return cost;
    }

    private boolean checkPointIsReached(int currentX, int endX, int currentY, int endY) {
        boolean forX = currentX == endX;
        boolean forY = currentY == endY;
        return forX && forY;
    }
}
