package pl.marek;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class DynamicProgrammingAlgorithm {

    private Integer[][][] grid;
    private Integer[][][] memo;

    DynamicProgrammingAlgorithm(Integer[][][] grid) {
        this.grid = grid;
        this.memo = new Integer[grid.length][grid.length][2];
    }

    Integer solve(Integer X, Integer Y) {
        System.out.println("DP - Calculate path to point : " + "[ " + X + ", " + Y + " ]" );
        Integer cost = dynamicSolver(X, Y);
        showPath(X, Y);
        System.out.println("Cost : " + cost);
        System.out.println();
        return cost;
    }

    private Integer dynamicSolver(Integer X, Integer Y) {
        if (X == 0 && Y == 0) {
            memo[0][0][0] = 0;
            memo[0][0][1] = 0;
            return 0;
        }

        if (memo[X][Y][0] != null && memo[X][Y][1] != null) {
            if (memo[X][Y][0] > memo[X][Y][1]) {
                return memo[X][Y][1];
            } else {
                return memo[X][Y][0];
            }
        }

        if (X == 0) {
            Integer cost = dynamicSolver(X, Y - 1) + grid[Y - 1][X][0];
            memo[X][Y][1] = cost;
            return memo[X][Y][1];
        }

        if (Y == 0) {
            Integer cost = dynamicSolver(X - 1, Y) + grid[Y][X - 1][1];
            memo[X][Y][0] = cost;
            return memo[X][Y][0];
        }

        Integer costFromLeft = dynamicSolver(X - 1, Y) + grid[Y][X - 1][1];
        Integer costFromBottom = dynamicSolver(X, Y - 1) + grid[Y - 1][X][0];
        memo[X][Y][1] = costFromBottom;
        memo[X][Y][0] = costFromLeft;

        if(costFromBottom < costFromLeft) {
            return costFromBottom;
        } else {
            return costFromLeft;
        }
    }

    private void showPath(Integer X, Integer Y) {
        List<String> paths = new ArrayList<>();
        int i = X;
        int j = Y;
        paths.add("[" + i + ", " + j + "]");

        while (memo[i][j][0] != 0 || memo[i][j][1] != 0) {
            if (memo[i][j][0] == null || memo[i][j][0] == 0) {
                j = j - 1;
            } else if (memo[i][j][1] == null || memo[i][j][1] == 0) {
                i = i - 1;
            } else if (memo[i][j][0] < memo[i][j][1]) {
                i = i - 1;
            } else if (memo[i][j][0] >= memo[i][j][1]) {
                j = j - 1;
            }
            paths.add("[" + i + ", " + j + "]");
        }
        Collections.reverse(paths);
        paths.forEach(System.out::println);
    }

    void printMemo() {
        System.out.println("--- Shortest paths ---");
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid.length; j++) {
                System.out.print("[" + memo[i][j][0] + " " + memo[i][j][1] + "] ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
