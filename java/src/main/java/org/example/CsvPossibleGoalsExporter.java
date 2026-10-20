package org.example;

import com.google.ortools.Loader;
import com.google.ortools.linearsolver.MPConstraint;
import com.google.ortools.linearsolver.MPSolver;
import com.google.ortools.linearsolver.MPVariable;
import put.dea.robustness.CCRPreferenceRelations;
import put.dea.robustness.ProblemData;
import tech.tablesaw.api.Table;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Pairwise POS and simultaneous ratio preference, with nonnegative unrestricted CCR weights. */
public class CsvPossibleGoalsExporter {
    private static final double TOL = 1e-9;
    private record Result(double score, String inputWeights, String outputWeights) {}

    public static void main(String[] args) throws Exception {
        if (args.length != 4) throw new IllegalArgumentException("reference.csv candidates.csv top_targets.csv output.csv");
        Loader.loadNativeLibraries();
        Table reference = Table.read().csv(args[0]);
        Table candidates = Table.read().csv(args[1]);
        Table targets = Table.read().csv(args[2]);
        List<String> inputs = columns(reference, "i"), outputs = columns(reference, "o");
        validate(reference, inputs, outputs);
        validate(candidates, inputs, outputs);
        List<Integer> top = new ArrayList<>();
        for (String name : targets.stringColumn("name")) {
            int index = reference.stringColumn("name").asList().indexOf(name);
            if (index < 0 || top.contains(index)) throw new IllegalArgumentException("Invalid/duplicate top target: " + name);
            top.add(index);
        }
        if (top.isEmpty()) throw new IllegalArgumentException("Empty common goal");
        Path output = Path.of(args[3]);
        Files.createDirectories(output.toAbsolutePath().getParent());
        CCRPreferenceRelations relations = new CCRPreferenceRelations();
        try (BufferedWriter writer = Files.newBufferedWriter(output);
             BufferedWriter witnesses = Files.newBufferedWriter(Path.of(args[3] + ".witnesses.csv"))) {
            writer.write("name,possible_over_refs,pairwise_top,joint_top,joint_score,joint_input_weights,joint_output_weights\n");
            witnesses.write("name,target,pairwise_possible,score,input_weights,output_weights\n");
            for (int row = 0; row < candidates.rowCount(); row++) {
                ProblemData data = problem(reference, candidates, row, inputs, outputs);
                List<String> possible = new ArrayList<>();
                boolean pairwiseTop = true;
                for (int j = 0; j < reference.rowCount(); j++) {
                    boolean pos = relations.isPossiblyPreferred(data, reference.rowCount(), j);
                    String target = reference.stringColumn("name").get(j);
                    if (pos) possible.add(target);
                    if (top.contains(j)) {
                        pairwiseTop &= pos;
                        Result single = solve(reference, candidates, row, List.of(j), inputs, outputs);
                        if (pos != (single.score >= 1-TOL)) throw new IllegalStateException("Pairwise library/LP disagreement near threshold: " + target);
                        witnesses.write(csv(candidates.stringColumn("name").get(row)) + "," + csv(target) + "," + pos + "," + single.score
                                + "," + csv(single.inputWeights) + "," + csv(single.outputWeights) + "\n");
                    }
                }
                Result joint = solve(reference, candidates, row, top, inputs, outputs);
                boolean jointTop = joint.score >= 1-TOL;
                if (jointTop && !pairwiseTop) throw new IllegalStateException("Joint goal must imply pairwise goal");
                writer.write(csv(candidates.stringColumn("name").get(row)) + "," + csv(String.join("|", possible)) + ","
                        + pairwiseTop + "," + jointTop + "," + joint.score + "," + csv(joint.inputWeights) + "," + csv(joint.outputWeights) + "\n");
            }
        }
        System.out.println("Evaluated both POS goals for " + candidates.rowCount() + " states against " + top.size() + " common top targets.");
    }

    private static Result solve(Table reference, Table candidates, int row, List<Integer> targets,
                                List<String> inputs, List<String> outputs) {
        MPSolver solver = MPSolver.createSolver("GLOP");
        if (solver == null) throw new IllegalStateException("GLOP unavailable");
        try {
            solver.setSolverSpecificParametersAsString("primal_feasibility_tolerance: 1e-10\ndual_feasibility_tolerance: 1e-10");
            double inf = MPSolver.infinity();
            MPVariable[] v = solver.makeNumVarArray(inputs.size(), 0, inf);
            MPVariable[] u = solver.makeNumVarArray(outputs.size(), 0, inf);
            MPConstraint inputNormalization = solver.makeConstraint(1, 1);
            MPConstraint candidateCap = solver.makeConstraint(-inf, 1);
            for (int i = 0; i < v.length; i++) inputNormalization.setCoefficient(v[i], candidates.numberColumn(inputs.get(i)).getDouble(row));
            for (int r = 0; r < u.length; r++) {
                double value = candidates.numberColumn(outputs.get(r)).getDouble(row);
                candidateCap.setCoefficient(u[r], value);
                solver.objective().setCoefficient(u[r], value);
            }
            // At optimum 1, candidate has ratio 1 and every target has ratio <= 1 with these SAME weights.
            for (int j : targets) {
                MPConstraint constraint = solver.makeConstraint(0, inf);
                for (int i = 0; i < v.length; i++) constraint.setCoefficient(v[i], reference.numberColumn(inputs.get(i)).getDouble(j));
                for (int r = 0; r < u.length; r++) constraint.setCoefficient(u[r], -reference.numberColumn(outputs.get(r)).getDouble(j));
            }
            solver.objective().setMaximization();
            if (solver.solve() != MPSolver.ResultStatus.OPTIMAL || !solver.verifySolution(1e-9, true))
                throw new IllegalStateException("Nonoptimal/unverified joint POS LP");
            List<String> inputWeights = new ArrayList<>(), outputWeights = new ArrayList<>();
            for (int i = 0; i < v.length; i++) inputWeights.add(inputs.get(i) + "=" + v[i].solutionValue());
            for (int r = 0; r < u.length; r++) outputWeights.add(outputs.get(r) + "=" + u[r].solutionValue());
            return new Result(solver.objective().value(), String.join("|", inputWeights), String.join("|", outputWeights));
        } finally {
            solver.delete();
        }
    }

    private static ProblemData problem(Table reference, Table candidates, int row, List<String> inputs, List<String> outputs) {
        double[][] x = new double[reference.rowCount()+1][inputs.size()];
        double[][] y = new double[reference.rowCount()+1][outputs.size()];
        for (int j = 0; j <= reference.rowCount(); j++) {
            Table table = j == reference.rowCount() ? candidates : reference;
            int index = j == reference.rowCount() ? row : j;
            for (int i = 0; i < inputs.size(); i++) x[j][i] = table.numberColumn(inputs.get(i)).getDouble(index);
            for (int r = 0; r < outputs.size(); r++) y[j][r] = table.numberColumn(outputs.get(r)).getDouble(index);
        }
        return new ProblemData(x, y, inputs, outputs);
    }

    private static List<String> columns(Table table, String prefix) {
        return table.columnNames().stream().filter(c -> c.matches(prefix + "\\d+"))
                .sorted(Comparator.comparingInt(c -> Integer.parseInt(c.substring(1)))).toList();
    }

    private static void validate(Table table, List<String> inputs, List<String> outputs) {
        if (inputs.isEmpty() || outputs.isEmpty() || table.rowCount() == 0 || table.stringColumn("name").unique().size() != table.rowCount())
            throw new IllegalArgumentException("Nonempty IO and unique names required");
        List<String> columns = new ArrayList<>(inputs);
        columns.addAll(outputs);
        for (int row = 0; row < table.rowCount(); row++) {
            String name = table.stringColumn("name").get(row);
            if (name.isBlank() || name.contains("|") || name.equals("[]")) throw new IllegalArgumentException("Invalid serialized name");
            for (String column : columns) {
                double value = table.numberColumn(column).getDouble(row);
                if (!Double.isFinite(value) || value <= 0) throw new IllegalArgumentException("Positive finite IO required");
            }
        }
    }

    private static String csv(String value) { return "\"" + value.replace("\"", "\"\"") + "\""; }
}
