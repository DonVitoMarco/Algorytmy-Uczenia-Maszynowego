package pl.marek;

public class App {

    public static void main(String[] args) {
        Integer[][][] grid = {
                {{6, 4}, {4, 1}, {3, 2}, {3, 3}, {2, null}},
                {{10, 3}, {4, 2}, {2, 2}, {3, 2}, {2, null}},
                {{20, 2}, {8, 3}, {3, 2}, {2, 1}, {3, null}},
                {{25, 1}, {10, 2}, {4, 2}, {3, 2}, {2, null}},
                {{null, 3}, {null, 3}, {null, 3}, {null, 2}, {null, null}}
        };

        DynamicProgrammingAlgorithm dynamicProgrammingAlgorithm = new DynamicProgrammingAlgorithm(grid);
        GreedyAlgorithm greedyAlgorithm = new GreedyAlgorithm(grid);

        greedyAlgorithm.solve(4, 4);
        dynamicProgrammingAlgorithm.solve(4, 4);
        dynamicProgrammingAlgorithm.printMemo();

        greedyAlgorithm.solve(1, 4);
        dynamicProgrammingAlgorithm.solve(1, 4);

        greedyAlgorithm.solve(3, 3);
        dynamicProgrammingAlgorithm.solve(3, 3);

        greedyAlgorithm.solve(4, 3);
        dynamicProgrammingAlgorithm.solve(4, 3);

        greedyAlgorithm.solve(2, 2);
        dynamicProgrammingAlgorithm.solve(2, 2);
    }
}
