function toggleMenu(){
  const m = document.getElementById('mobileMenu');
  if(!m) return;
  const isOpen = m.style.display === 'block';
  m.style.display = isOpen ? 'none' : 'block';
}

document.getElementById('year').textContent = new Date().getFullYear();
