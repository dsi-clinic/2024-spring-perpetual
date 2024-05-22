import { Card, CardBody } from "@nextui-org/react";

export default function Stat({ value, units }) {
  return (
    <Card className="max-w-[100px] bg-blue-50">
      <CardBody className="text-center flex flex-col items-center">
        <p>{units}</p>
        <h4 className="text-lg font-bold">{value}</h4>
      </CardBody>
    </Card>
  );
}
