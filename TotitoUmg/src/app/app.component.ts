import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

interface GameState {
  tablero: string[];
  esTurnoUsuario: boolean;
  finJuego: boolean;
  hayGanador: string | null;
  historicoMovPC: Map<string, number>;
  seguimientoTablero: string[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  estadoJuego: GameState = this.obtenerEstadoInicial();
  juegosJugados = 0;
  usuarioGanados = 0;
  pcGanados = 0;
  empates = 0;


  nuevoJuego(): void {
    this.estadoJuego = this.obtenerEstadoInicial();
    this.cargarStorage();
  }

  obtenerEstadoInicial(): GameState {
    return {
      tablero: Array(9).fill(''),
      esTurnoUsuario: true,
      finJuego: false,
      hayGanador: null,
      historicoMovPC: new Map(),
      seguimientoTablero: []
    };
  }

  get EstadoJuego(): string {
    if (this.estadoJuego.finJuego) {
      if (this.estadoJuego.hayGanador) {
        return `¡Gana ${this.estadoJuego.hayGanador === 'X' ? 'Usuario' : 'PC'}!`;
      }
      return '¡Empate!';
    }
    return this.estadoJuego.esTurnoUsuario ? 'Turno usuario' : 'Turno PC';
  }


  mueveUsuario(index: number): void {
    if (
      this.estadoJuego.finJuego ||
      !this.estadoJuego.esTurnoUsuario ||
      this.estadoJuego.tablero[index] !== ''
    ) {
      return;
    }

    this.estadoJuego.tablero[index] = 'X';
    this.verificaEstadoJuego();

    if (!this.estadoJuego.finJuego) {
      this.estadoJuego.esTurnoUsuario = false;
      setTimeout(() => this.muevePC(), 500);
    }
  }

  muevePC(): void {
    let mejorMovimiento = -1;
    let mejorPuntuacion = -Infinity;

    // obtener movimientos disponibles
    const movimientosDisponibles = this.estadoJuego.tablero
      .map((cell, index) => cell === '' ? index : -1)
      .filter(index => index !== -1);

    // Elejir movimiento basado en aprendizaje o aleatorio si no hay movimientos aprendidos
    for (const movimiento of movimientosDisponibles) {
      const tableroTmp = [...this.estadoJuego.tablero];
      tableroTmp[movimiento] = 'O';
      let idMov = '';
      for (let i = 0; i < tableroTmp.length; i++) {
        if (tableroTmp[i] === '') {
          idMov += '-';
        }
        else {
          idMov += tableroTmp[i];
        }
      }

      const puntuacion = this.estadoJuego.historicoMovPC.get(idMov) || 0;

      if (puntuacion > mejorPuntuacion) {
        mejorPuntuacion = puntuacion;
        mejorMovimiento = movimiento;
      }
    }

    // Si no se encuentra el mejor movimiento, elige aleatorio
    if (mejorMovimiento === -1) {
      mejorMovimiento = movimientosDisponibles[Math.floor(Math.random() * movimientosDisponibles.length)];
    }

    this.estadoJuego.tablero[mejorMovimiento] = 'O';
    this.verificaEstadoJuego();
    this.estadoJuego.esTurnoUsuario = true;

    ////////
    let movElejidoTablero = '';
    for (let j = 0; j < this.estadoJuego.tablero.length; j++) {
      if (this.estadoJuego.tablero[j] === '') {
        movElejidoTablero += '-';
      }
      else {
        movElejidoTablero += this.estadoJuego.tablero[j];
      }
    }
    this.estadoJuego.seguimientoTablero.push(movElejidoTablero);
  }


  verificaEstadoJuego(): void {
    const winPatterns = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
      [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
      [0, 4, 8], [2, 4, 6] // Diagonals
    ];

    for (const pattern of winPatterns) {
      const [a, b, c] = pattern;
      if (
        this.estadoJuego.tablero[a] &&
        this.estadoJuego.tablero[a] === this.estadoJuego.tablero[b] &&
        this.estadoJuego.tablero[a] === this.estadoJuego.tablero[c]
      ) {
        this.estadoJuego.finJuego = true;
        this.estadoJuego.hayGanador = this.estadoJuego.tablero[a];
        this.aprendizaje();
        this.actualizaEstadisticas();
        return;
      }
    }

    if (!this.estadoJuego.tablero.includes('')) {
      this.estadoJuego.finJuego = true;
      this.estadoJuego.hayGanador = null;
      this.actualizaEstadisticas();
      return;
    }
  }


  aprendizaje(): void {
    const boardStates = this.estadoJuego.seguimientoTablero;
    const reward = this.estadoJuego.hayGanador === 'O' ? 1 : this.estadoJuego.hayGanador === 'X' ? -1 : 0;

    for (const state of boardStates) {
      const currentScore = this.estadoJuego.historicoMovPC.get(state) || 0;
      this.estadoJuego.historicoMovPC.set(state, currentScore + reward);
    }
  }

  obtenerCasilla(index: number): string {
    if (
      this.estadoJuego.finJuego ||
      !this.estadoJuego.esTurnoUsuario ||
      this.estadoJuego.tablero[index] !== ''
    ) {
      return 'not-allowed';
    }
    return 'pointer';
  }

  actualizaEstadisticas(): void {
    this.juegosJugados++;
    if (this.estadoJuego.hayGanador === 'X') {
      this.usuarioGanados++;
    } else if (this.estadoJuego.hayGanador === 'O') {
      this.pcGanados++;
    } else {
      this.empates++;
    }
    this.guardarStorage();
  }

  guardarStorage(): void {
    const stats = {
      juegosJugados: this.juegosJugados,
      usuarioGanados: this.usuarioGanados,
      pcGanados: this.pcGanados,
      empates: this.empates,
      historicoMovPC: Array.from(this.estadoJuego.historicoMovPC.entries())
    };
    localStorage.setItem('totito_stats', JSON.stringify(stats));
  }

  cargarStorage(): void {
    const cargaEstadisticas = localStorage.getItem('totito_stats');
    if (cargaEstadisticas) {
      const estadistica = JSON.parse(cargaEstadisticas);
      this.juegosJugados = estadistica.juegosJugados;
      this.usuarioGanados = estadistica.usuarioGanados;
      this.pcGanados = estadistica.pcGanados;
      this.empates = estadistica.empates;
      this.estadoJuego.historicoMovPC = new Map(estadistica.historicoMovPC);
    }
  }
}
