import { render, screen } from '@testing-library/react';
import App from './App';

test('renders AI interview text', () => {
  render(<App />);
  const titleElement = screen.getByText(/AI-Based human interview/i);
  expect(titleElement).toBeInTheDocument();
});
