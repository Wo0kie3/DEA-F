package org.example;

import put.dea.robustness.CCRExtremeEfficiencies;
import put.dea.robustness.CCRExtremeRanks;
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

public class CsvCandidateRobustMetricsExporter {

    public static void main(String[] args) throws IOException {
        String referenceCsvPath = args.length > 0 ? args[0] : "../input/airports.csv";
        String candidatesCsvPath = args.length > 1 ? args[1] : "../output/fictive_candidates.csv";
        String outputCsvPath = args.length > 2 ? args[2] : "../output/fictive_candidate_metrics.csv";

        new CsvCandidateRobustMetricsExporter().run(referenceCsvPath, candidatesCsvPath, outputCsvPath);

        System.out.println("Done. Candidate robust metrics saved to: " + outputCsvPath);
    }

    public void run(
            String referenceCsvPath,
            String candidatesCsvPath,
            String outputCsvPath
    ) throws IOException {
        List<DmuRow> referenceRows = readRowsFromCsv(referenceCsvPath);
        List<DmuRow> candidates = readRowsFromCsv(candidatesCsvPath);

        if (referenceRows.isEmpty()) {
            throw new IllegalArgumentException("Reference CSV contains no data rows: " + referenceCsvPath);
        }
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("Candidates CSV contains no data rows: " + candidatesCsvPath);
        }

        List<String> inputNames = detectColumns(referenceRows.get(0).rawValues().keySet(), "i");
        List<String> outputNames = detectColumns(referenceRows.get(0).rawValues().keySet(), "o");
        List<String> referenceNames = referenceRows.stream().map(DmuRow::name).toList();

        CCRExtremeEfficiencies extremeEfficiencies = new CCRExtremeEfficiencies();
        CCRExtremeRanks extremeRanks = new CCRExtremeRanks();
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
                List<DmuRow> evaluationRows = new ArrayList<>(referenceRows);
                evaluationRows.add(candidate);

                ProblemDataBundle bundle = buildProblemData(evaluationRows, inputNames, outputNames);
                int candidateIdx = bundle.dmuNames().size() - 1;

                double bestEfficiency = extremeEfficiencies.maxEfficiency(bundle.problemData(), candidateIdx);
                double worstEfficiency = extremeEfficiencies.minEfficiency(bundle.problemData(), candidateIdx);
                int bestRank = extremeRanks.minRank(bundle.problemData(), candidateIdx);
                int worstRank = extremeRanks.maxRank(bundle.problemData(), candidateIdx);

                List<String> candidateNecessaryOver = new ArrayList<>();
                List<String> candidatePossibleOver = new ArrayList<>();
                List<String> referenceNecessaryOverCandidate = new ArrayList<>();
                List<String> referencePossibleOverCandidate = new ArrayList<>();

                for (int refIdx = 0; refIdx < referenceRows.size(); refIdx++) {
                    String refName = referenceNames.get(refIdx);

                    if (preferenceRelations.isNecessarilyPreferred(bundle.problemData(), candidateIdx, refIdx)) {
                        candidateNecessaryOver.add(refName);
                    }
                    if (preferenceRelations.isPossiblyPreferred(bundle.problemData(), candidateIdx, refIdx)) {
                        candidatePossibleOver.add(refName);
                    }
                    if (preferenceRelations.isNecessarilyPreferred(bundle.problemData(), refIdx, candidateIdx)) {
                        referenceNecessaryOverCandidate.add(refName);
                    }
                    if (preferenceRelations.isPossiblyPreferred(bundle.problemData(), refIdx, candidateIdx)) {
                        referencePossibleOverCandidate.add(refName);
                    }
                }

                writeRow(
                        writer,
                        candidate,
                        bestEfficiency,
                        worstEfficiency,
                        bestRank,
                        worstRank,
                        candidateNecessaryOver,
                        candidatePossibleOver,
                        referenceNecessaryOverCandidate,
                        referencePossibleOverCandidate
                );
            }

            System.out.printf("Progress: 100%% (%d/%d) | ETA: 0.0 sec%n", total, total);
        }
    }

    private void writeHeader(BufferedWriter writer, Set<String> candidateColumns) throws IOException {
        List<String> cols = new ArrayList<>(candidateColumns);
        cols.add("best_efficiency");
        cols.add("worst_efficiency");
        cols.add("best_rank");
        cols.add("worst_rank");
        cols.add("score_width");
        cols.add("rank_width");
        cols.add("candidate_necessary_over_count");
        cols.add("candidate_possible_over_count");
        cols.add("reference_necessary_over_candidate_count");
        cols.add("reference_possible_over_candidate_count");
        cols.add("candidate_necessary_over_refs");
        cols.add("candidate_possible_over_refs");
        cols.add("reference_necessary_over_candidate_refs");
        cols.add("reference_possible_over_candidate_refs");

        writer.write(String.join(",", cols));
        writer.newLine();
    }

    private void writeRow(
            BufferedWriter writer,
            DmuRow candidate,
            double bestEfficiency,
            double worstEfficiency,
            int bestRank,
            int worstRank,
            List<String> candidateNecessaryOver,
            List<String> candidatePossibleOver,
            List<String> referenceNecessaryOverCandidate,
            List<String> referencePossibleOverCandidate
    ) throws IOException {
        List<String> cols = new ArrayList<>();
        for (String col : candidate.rawValues().keySet()) {
            cols.add(escapeCsv(candidate.rawValues().get(col)));
        }

        cols.add(formatDouble(bestEfficiency));
        cols.add(formatDouble(worstEfficiency));
        cols.add(Integer.toString(bestRank));
        cols.add(Integer.toString(worstRank));
        cols.add(formatDouble(bestEfficiency - worstEfficiency));
        cols.add(Integer.toString(worstRank - bestRank));
        cols.add(Integer.toString(candidateNecessaryOver.size()));
        cols.add(Integer.toString(candidatePossibleOver.size()));
        cols.add(Integer.toString(referenceNecessaryOverCandidate.size()));
        cols.add(Integer.toString(referencePossibleOverCandidate.size()));
        cols.add(escapeCsv(String.join("|", candidateNecessaryOver)));
        cols.add(escapeCsv(String.join("|", candidatePossibleOver)));
        cols.add(escapeCsv(String.join("|", referenceNecessaryOverCandidate)));
        cols.add(escapeCsv(String.join("|", referencePossibleOverCandidate)));

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

    private String formatDouble(double value) {
        return String.format(Locale.US, "%.10f", value);
    }

    private String escapeCsv(String value) {
        if (value == null) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("|")) {
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
