package org.example;

import put.dea.robustness.CCRPreferenceRelations;
import put.dea.robustness.ProblemData;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class CsvPairwisePossibleCandidateEvaluator {

    public static void main(String[] args) throws IOException {
        String referenceCsvPath = args.length > 0 ? args[0] : "../output/reference.csv";
        String candidatesCsvPath = args.length > 1 ? args[1] : "../output/candidates.csv";
        String outputCsvPath = args.length > 2 ? args[2] : "../output/pairwise_possible_results.csv";

        new CsvPairwisePossibleCandidateEvaluator().run(
                referenceCsvPath,
                candidatesCsvPath,
                outputCsvPath
        );

        System.out.println("Done. Results saved to: " + outputCsvPath);
    }

    public void run(
            String referenceCsvPath,
            String candidatesCsvPath,
            String outputCsvPath
    ) throws IOException {
        List<DmuRow> referenceRows = readRowsFromCsv(referenceCsvPath);
        List<DmuRow> candidates = readRowsFromCsv(candidatesCsvPath);

        if (referenceRows.size() != 1) {
            throw new IllegalArgumentException(
                    "Reference CSV must contain exactly one DMU row, got " + referenceRows.size()
            );
        }
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("Candidates CSV contains no data rows: " + candidatesCsvPath);
        }

        DmuRow reference = referenceRows.get(0);
        List<String> inputNames = detectColumns(reference.rawValues().keySet(), "i");
        List<String> outputNames = detectColumns(reference.rawValues().keySet(), "o");

        System.out.println("==================================================");
        System.out.println("PAIRWISE POSSIBLE CANDIDATE EVALUATION");
        System.out.println("Reference DMU: " + reference.name());
        System.out.println("Candidates file: " + candidatesCsvPath);
        System.out.println("Total candidates: " + candidates.size());
        System.out.println("Inputs: " + String.join(", ", inputNames));
        System.out.println("Outputs: " + String.join(", ", outputNames));
        System.out.println("==================================================");

        CCRPreferenceRelations preferenceRelations = new CCRPreferenceRelations();

        Path outputPath = Path.of(outputCsvPath);
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            writeHeader(writer, candidates.get(0).rawValues().keySet());

            long startTime = System.currentTimeMillis();
            int total = candidates.size();
            int lastPct = -1;

            for (int i = 0; i < total; i++) {
                int pct = (int) ((i * 100.0) / total);

                if (pct != lastPct && pct % 5 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double perItem = elapsed / (double) (i + 1);
                    long remaining = (long) (perItem * (total - i));

                    System.out.printf(
                            "Progress: %d%% (%d/%d) | ETA: %.1f sec%n",
                            pct,
                            i,
                            total,
                            remaining / 1000.0
                    );

                    lastPct = pct;
                }

                DmuRow candidate = candidates.get(i);
                List<DmuRow> pair = List.of(candidate, reference);
                ProblemDataBundle bundle = buildProblemData(pair, inputNames, outputNames);

                List<List<Boolean>> necessaryRelations =
                        preferenceRelations.checkNecessaryPreferenceForAll(bundle.problemData());
                List<List<Boolean>> possibleRelations =
                        preferenceRelations.checkPossiblePreferenceForAll(bundle.problemData());

                int candidateIdx = bundle.dmuNames().indexOf(candidate.name());
                int referenceIdx = bundle.dmuNames().indexOf(reference.name());

                writeRow(
                        writer,
                        candidate,
                        reference.name(),
                        possibleRelations.get(candidateIdx).get(referenceIdx),
                        possibleRelations.get(referenceIdx).get(candidateIdx),
                        necessaryRelations.get(candidateIdx).get(referenceIdx),
                        necessaryRelations.get(referenceIdx).get(candidateIdx)
                );
            }

            System.out.printf("Progress: 100%% (%d/%d) | ETA: 0.0 sec%n", total, total);
        }
    }

    private void writeHeader(
            BufferedWriter writer,
            Set<String> candidateColumns
    ) throws IOException {
        List<String> cols = new ArrayList<>(candidateColumns);
        cols.add("reference_name");
        cols.add("candidate_over_reference_possible");
        cols.add("reference_over_candidate_possible");
        cols.add("candidate_over_reference_necessary");
        cols.add("reference_over_candidate_necessary");

        writer.write(String.join(",", cols));
        writer.newLine();
    }

    private void writeRow(
            BufferedWriter writer,
            DmuRow candidate,
            String referenceName,
            boolean candidateOverReferencePossible,
            boolean referenceOverCandidatePossible,
            boolean candidateOverReferenceNecessary,
            boolean referenceOverCandidateNecessary
    ) throws IOException {
        List<String> cols = new ArrayList<>();

        for (String col : candidate.rawValues().keySet()) {
            cols.add(escapeCsv(candidate.rawValues().get(col)));
        }

        cols.add(escapeCsv(referenceName));
        cols.add(Boolean.toString(candidateOverReferencePossible));
        cols.add(Boolean.toString(referenceOverCandidatePossible));
        cols.add(Boolean.toString(candidateOverReferenceNecessary));
        cols.add(Boolean.toString(referenceOverCandidateNecessary));

        writer.write(String.join(",", cols));
        writer.newLine();
    }

    private ProblemDataBundle buildProblemData(
            List<DmuRow> rows,
            List<String> inputNames,
            List<String> outputNames
    ) {
        double[][] inputs = new double[rows.size()][inputNames.size()];
        double[][] outputs = new double[rows.size()][outputNames.size()];
        List<String> dmuNames = new ArrayList<>();

        for (int r = 0; r < rows.size(); r++) {
            DmuRow row = rows.get(r);
            dmuNames.add(row.name());

            for (int c = 0; c < inputNames.size(); c++) {
                inputs[r][c] = parseDouble(row.rawValues().get(inputNames.get(c)), row.name(), inputNames.get(c));
            }

            for (int c = 0; c < outputNames.size(); c++) {
                outputs[r][c] = parseDouble(row.rawValues().get(outputNames.get(c)), row.name(), outputNames.get(c));
            }
        }

        ProblemData data = new ProblemData(inputs, outputs, inputNames, outputNames);
        return new ProblemDataBundle(data, dmuNames);
    }

    private List<DmuRow> readRowsFromCsv(String csvPath) throws IOException {
        List<DmuRow> rows = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Path.of(csvPath))) {
            String headerLine = reader.readLine();
            if (headerLine == null || headerLine.isBlank()) {
                throw new IllegalArgumentException("CSV is empty: " + csvPath);
            }

            String[] headers = splitCsvLine(headerLine);

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }

                String[] parts = splitCsvLine(line);
                if (parts.length != headers.length) {
                    throw new IllegalArgumentException(
                            "Invalid CSV row. Expected " + headers.length + " columns but got " + parts.length +
                                    ". Row: " + line
                    );
                }

                Map<String, String> rawValues = new LinkedHashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    rawValues.put(headers[i].trim(), unquote(parts[i].trim()));
                }

                String name = rawValues.get("name");
                if (name == null || name.isBlank()) {
                    throw new IllegalArgumentException("Encountered row with empty 'name'.");
                }

                rows.add(new DmuRow(name, rawValues));
            }
        }

        return rows;
    }

    private List<String> detectColumns(Set<String> allColumns, String prefix) {
        return allColumns.stream()
                .filter(col -> col.matches("^" + prefix + "\\d+$"))
                .sorted(Comparator.comparingInt(col -> Integer.parseInt(col.substring(1))))
                .collect(Collectors.toList());
    }

    private double parseDouble(String value, String dmuName, String colName) {
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid numeric value for DMU='" + dmuName + "', column='" + colName + "': " + value, e
            );
        }
    }

    private String escapeCsv(String value) {
        if (value == null) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private String unquote(String value) {
        String v = value;
        if (v.startsWith("\"") && v.endsWith("\"") && v.length() >= 2) {
            v = v.substring(1, v.length() - 1).replace("\"\"", "\"");
        }
        return v;
    }

    private String[] splitCsvLine(String line) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);

            if (ch == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (ch == ',' && !inQuotes) {
                tokens.add(current.toString());
                current.setLength(0);
            } else {
                current.append(ch);
            }
        }

        tokens.add(current.toString());
        return tokens.toArray(new String[0]);
    }

    private record DmuRow(String name, Map<String, String> rawValues) {}
    private record ProblemDataBundle(ProblemData problemData, List<String> dmuNames) {}
}
