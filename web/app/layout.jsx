import './style.css';
export const metadata = {
  title: 'Hands-on WebAssembly',
  description:
    'Seven interactive machine learning assignments, running in your browser.',
};
export default function Layout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
