
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { missingRatioByYear } from "@/utils/thailandData";

const YearlyMissingRatioCard = () => {
  return (
    <Card className="h-full flex flex-col shadow-none bg-green-50/50 border border-green-100">
      <CardHeader>
        <CardTitle className="text-2xl text-green-700">Average Missing Value Ratio by Year</CardTitle>
        <CardDescription>
          Showing the reduction in missing data and justification for the 1950 cutoff
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-2 pb-4">
        <div className="h-[232px] w-full">
          <ResponsiveContainer>
            <LineChart data={missingRatioByYear} margin={{ top: 10, right: 24, left: 4, bottom: 24 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="year"
                fontSize={11}
                minTickGap={12}
                angle={-40}
                textAnchor="end"
                interval={6}
                height={44}
                tick={{ dy: 8 }}
                tickFormatter={v => v}
              />
              <YAxis
                tickFormatter={v => `${Math.round((v as number) * 100)}%`}
                domain={[0, 1]}
                width={38}
              />
              <Tooltip formatter={(v:number) => `${(v as number * 100).toFixed(1)}%`} />
              <Line
                type="monotone"
                dataKey="missing_ratio"
                stroke="#6366f1"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: "#6366f1" }}
                isAnimationActive={false}
              />
              <ReferenceLine
                x={1950}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{
                  value: "1950 cutoff",
                  position: "bottom",
                  fill: "#ef4444",
                  fontSize: 11,
                  dy: 8
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default YearlyMissingRatioCard;
