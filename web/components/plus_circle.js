export default function PlusCircleIcon({
  fillColor = "none",
  strokeColor = "currentColor",
  strokeWidth = 3,
  size,
  height,
  width,
  label,
  ...props
}) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size || width || 24}
      height={size || height || 24}
      fill={fillColor}
      viewBox="0 0 24 24"
      strokeWidth={strokeWidth}
      stroke={strokeColor}
      className="w-6 h-6"
      {...props}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
    </svg>
  );
}
